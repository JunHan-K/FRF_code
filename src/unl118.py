"""Shared builder for PREREG117-119 (unlearning arms by spec string, held-out selection signals, strong attack).
Arm specs:
  native | HRr | RL | SCRUB
  SDa<alpha>        selective dampening on all 12 blocks (threshold alpha, lambda 1), dog head rows re-initialised, dog-free
                    rank-4 LoRA repair (3,000 steps, masked teacher), folded  (= OURS with alpha 3 / 10)
  SD4a<alpha>       same, dampening only on the four blocks ranked highest by the FRF class-conditional magnitude ratio (p96.locate)
  SDw<gamma>        all 12 blocks, per-block threshold alpha_k = a0 * (mean ratio / ratio_k)^gamma (ratio = p96.locate on
                    id_fit[:2000]); a0 set by bisection so that the mean dampened fraction over block parameters equals that of
                    uniform dampening at the model's OURS alpha; then head reset + repair as above
Recipes of RL / SCRUB / repair / attack are those of PREREG104 / 108 / 106 (unchanged code)."""
import copy
import math
import os
import sys

import torch  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import p18  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import FORGET, locate  # noqa: E402
from p99 import wrap, fold  # noqa: E402
from p100 import fisher  # noqa: E402
from p101 import repair  # noqa: E402
from p102 import head_reinit  # noqa: E402
from p110 import frf  # noqa: E402
import select_ours as SEL  # noqa: E402

STEPS = int(os.environ.get("U118_STEPS", 3000))  # paper: default value (override only for quick tests)
# damage strength of OURS on dogs, chosen on id_hold only (see src/select_ours.py)
OURS_ALPHA = {m: float(a) for m, a in {**SEL.SELECTED["dogs"], **SEL.SELECTED_VITB}.items()}


@torch.no_grad()
def dampen_k(blocks_params, If, Ir, alphas):
    """per-block thresholds; blocks_params[k] = list of params of block k; returns the dampened fraction: the mean, across parameter tensors (unweighted), of the fraction
    of entries dampened in each tensor"""
    frac = []
    for k, ps in enumerate(blocks_params):
        for q, f_, r_ in zip(ps, If[k], Ir[k]):
            mask = f_ > alphas[k] * r_
            beta = (r_ / f_.clamp_min(1e-30)).clamp(max=1.0)
            q[mask] *= beta[mask]
            frac.append(float(mask.float().mean()))
    return sum(frac) / len(frac)


def frac_at(If, Ir, alphas):
    fr = []
    for k in range(len(If)):
        for f_, r_ in zip(If[k], Ir[k]):
            fr.append(float((f_ > alphas[k] * r_).float().mean()))
    return sum(fr) / len(fr)


def build(model, arm, nat, nrm, xd, yd, xr, yr, xf, yf, info):
    if arm == "native":
        return copy.deepcopy(nat)
    m = copy.deepcopy(nat)
    if arm == "RL":
        from p104 import train_full
        return p18.frozen(train_full(m, nrm, xr, yr, STEPS, xd=xd))
    if arm == "SCRUB":
        import p108
        p108.xd_labels = yd
        return p18.frozen(p108.train(m, nat, nrm, xd, xr, yr, "SCRUB", STEPS))
    if arm.startswith("SD"):
        bp = [list(m.blocks[k].parameters()) for k in range(12)]
        flat = [q for ps in bp for q in ps]
        If_, Ir_ = fisher(m, nrm, flat, xd[:400], yd[:400]), fisher(m, nrm, flat, xr[:800], yr[:800])
        sizes = [len(ps) for ps in bp]
        If, Ir, i = [], [], 0
        for n in sizes:
            If.append(If_[i:i + n]); Ir.append(Ir_[i:i + n]); i += n
        if arm.startswith("SDa"):
            al = [float(arm[3:])] * 12
        elif arm.startswith("SD4a"):
            ratio, _ = locate(nat, nrm, xf[:2000], yf[:2000])
            top = sorted(range(12), key=lambda k: -ratio[k])[:4]
            info["blocks"] = sorted(top)
            al = [float(arm[4:]) if k in top else float("inf") for k in range(12)]
        elif arm.startswith("SDw"):
            gam = float(arm[3:])
            ratio, _ = locate(nat, nrm, xf[:2000], yf[:2000])
            r = torch.tensor(ratio, dtype=torch.float64)
            shape = ((r.mean() / r) ** gam).tolist()
            target = frac_at(If, Ir, [OURS_ALPHA[model]] * 12)
            lo, hi = math.log(1e-3), math.log(1e4)
            for _ in range(40):  # frac decreases with a0
                mid = 0.5 * (lo + hi)
                if frac_at(If, Ir, [math.exp(mid) * s for s in shape]) > target:
                    lo = mid
                else:
                    hi = mid
            a0 = math.exp(0.5 * (lo + hi))
            al = [a0 * s for s in shape]
            info.update(ratio=ratio, target_frac=target)
        else:
            raise ValueError(arm)
        info["alphas"] = [a if math.isfinite(a) else None for a in al]
        info["dampened_fraction"] = dampen_k(bp, If, Ir, al)
        del If_, Ir_, If, Ir
    elif arm != "HRr":
        raise ValueError(arm)
    head_reinit(m)
    m = p18.frozen(m)
    ps = wrap(m)
    repair(m, nat, nrm, xr, ps, STEPS, [])
    fold(m)
    return p18.frozen(m)


def dsets():
    """fit / hold / fresh splits with forget masks"""
    xf, yf = S.cache("id_fit")
    xh, yh = S.cache("id_hold")
    x, y = S.cache("fresh")
    return (xf, yf), (xh, yh), (x, y)


def Dstat(r):
    import statistics as st
    return st.median([f["nu"] - g["nu"] for f, g in zip(r["F"], r["R"])])


def hold_frf(nat, U, nrm, xh, yh):
    hf = torch.isin(yh, FORGET)
    xF, xR = xh[hf], xh[~hf]
    ri = torch.randperm(len(xR), generator=torch.Generator().manual_seed(0))[:len(xF)]
    out = dict(F=frf(nat, U, nrm, xF), R=frf(nat, U, nrm, xR[ri]))
    out["D"] = Dstat(out)
    return out


def accs(m, nrm, x, y):
    f = torch.isin(y, FORGET)
    return dict(forget=acc(m, nrm, x[f], y[f]), retain=acc(m, nrm, x[~f], y[~f]))
