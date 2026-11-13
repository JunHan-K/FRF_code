"""PREREG96 (GPU): unlearning by disruption and recovery; also provides the dog forget set FORGET and the block locator
used by the later unlearning scripts. Relative to the base design below: (1) the stopping rule requires BOTH forget top-1 on id_hold <= 0.1 x native AND retain top-1 on id_hold >= native - 0.05 (so recovery happens);
(2) weaker noise (0.25 / 0.5 / 0.75 x std); (3) block removal arms: R (the two FRF-located blocks removed), Rcfe (the first
two blocks of the Sec. 7 FRF-magnitude removal order), Rrand (two random blocks). Arms: U N025 N05 N075 L E EN025 R Rcfe Rrand.
Written 2026-11-07 10:00 KST, before any PREREG96 network exists.
--- base design ---
Unlearning by disruption and recovery. Forget set F = the 118 ImageNet dog classes (151-268).
Locate (selection images id_fit, never evaluation images): for every block k, the Fisher energy of its effect at its own slot
(native network, block k vs skipped) on forget-class images divided by the same on retain images (class-conditional FRF
magnitude); the two blocks with the largest ratio are 'located' (primary). Secondary locate: the two blocks whose skipping
lowers forget-class top-1 most relative to retain-class top-1 (damage).
Arms (then the same recovery: distillation from the native network with forget-class logits masked, lr S.LR, wd S.WD,
batch 64, seed 0; stop at the first 50-step check with forget top-1 on id_hold <= 0.1 x native, at most 3,000 steps):
  U       no disruption, all parameters trained (fine-tuning unlearning; = PREREG91 with the stopping rule)
  N05 N1 N2 N4  Gaussian noise sigma x std(tensor) added to every parameter of the two located blocks, all parameters trained
  L       the two located blocks swap slots, all parameters trained
  E       no disruption, only the two located blocks trained
  EN      N1 noise, only the two located blocks trained
  Nrand   N1 noise on two random blocks (generator seed 95), all parameters trained
  Ndmg    N1 noise on the two damage-located blocks, all parameters trained
Measured on fresh (labels): forget / retain top-1; steps to criterion; stitch(k) as PREREG91; relearning (rank-4 LoRA,
distillation from the native network on retain-class id_fit images only, 300 steps, lr 1e-3); frame of each block k at slot k
(nu, E, change energy) on forget images and on retain images[:500].
Written 2026-11-06 20:20 KST, before any PREREG95 network exists.
usage: p95.py <model> <arm>"""
import copy
import json
import math
import os
import sys

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p90 import slot_effects, frame  # noqa: E402
from p91 import acc  # noqa: E402

OUT = os.environ.get("P96_OUT", "out/p96")
MAXS = int(os.environ.get("P96_STEPS", 3000))  # P96_STEPS / P96_OUT: smoke test only
RSTEPS = int(os.environ.get("P96_RSTEPS", 300))
FORGET = torch.arange(151, 269)
SIG = dict(N025=0.25, N05=0.5, N075=0.75, EN025=0.25)
REM = ("R", "Rcfe", "Rrand")


class Masked(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        z = self.m(x).float().clone()
        z[:, FORGET.to(z.device)] = -1e4
        return z


@torch.no_grad()
def locate(nat, nrm, x, y):
    """class-conditional FRF magnitude ratio and forget-specific skip damage per block (selection images)"""
    isf = torch.isin(y, FORGET)
    en = {"f": torch.zeros(12, dtype=torch.float64), "r": torch.zeros(12, dtype=torch.float64)}
    ok = {"f": torch.zeros(12), "r": torch.zeros(12)}
    n = {"f": 0, "r": 0}
    okn = {"f": 0.0, "r": 0.0}
    for s in range(0, len(x), 50):
        xb = nrm(x[s:s + 50].to(S.dev).float().div_(255))
        yb, fb = y[s:s + 50], isf[s:s + 50]
        zf = nat(xb).float()
        p = zf.softmax(-1).double()
        for g, msk in (("f", fb), ("r", ~fb)):
            n[g] += int(msk.sum())
            okn[g] += float((zf.argmax(-1).cpu() == yb)[msk].sum())
        for k in range(12):
            h = e13.emb(nat, xb)
            for t, b in enumerate(nat.blocks):
                if t != k:
                    h = b(h)
            zs = nat.forward_head(nat.norm(h)).float()
            d = (zf - zs).double()
            e = ((p * d * d).sum(-1) - (p * d).sum(-1) ** 2).cpu()
            hit = (zs.argmax(-1).cpu() == yb).float()
            for g, msk in (("f", fb), ("r", ~fb)):
                en[g][k] += e[msk].sum()
                ok[g][k] += hit[msk].sum()
    ratio = (en["f"] / n["f"]) / (en["r"] / n["r"]).clamp_min(1e-30)
    dmg = (okn["f"] / n["f"] - ok["f"] / n["f"]) - (okn["r"] / n["r"] - ok["r"] / n["r"])
    return ratio.tolist(), dmg.tolist()


@torch.no_grad()
def stitch(stu, nat, nrm, x, y, k, kept, bs=100):
    """unlearned embedding and the unlearned blocks whose native index is < k, then native blocks[k:], native norm and head
    (PREREG91 stitch; with removed blocks the unlearned prefix simply lacks them)"""
    ok = 0
    pre = [stu.blocks[i] for i, t in enumerate(kept) if t < k]
    for s in range(0, len(x), bs):
        h = e13.emb(stu, nrm(x[s:s + bs].to(S.dev).float().div_(255)))
        for b in pre:
            h = b(h)
        for b in nat.blocks[k:]:
            h = b(h)
        ok += int((nat.forward_head(nat.norm(h)).argmax(-1).cpu() == y[s:s + bs]).sum())
    return ok / max(len(x), 1)


def unlearn(m, nat, nrm, xf, params, xh, yh, nat_forget, nat_retain):
    """masked-teacher distillation with the paper's recipe; stop at the first 50-step check with id_hold forget top-1 <= 0.1 x native"""
    for q in m.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    teacher = Masked(nat)
    opt = torch.optim.AdamW(params, lr=S.LR, weight_decay=S.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, MAXS) / MAXS)))
    g = torch.Generator().manual_seed(0)
    hf = torch.isin(yh, FORGET)
    trace = []
    for t in range(MAXS):
        m.train()
        idx = torch.randint(len(xf), (S.BS,), generator=g)
        x = nrm(S.augment(xf[idx].to(S.dev, non_blocking=True), g))
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            lps = m(x[s:s + S.MB]).log_softmax(-1)
            with torch.no_grad():
                lpt = teacher(x[s:s + S.MB]).log_softmax(-1)
            ((lpt.exp() * (lpt - lps)).sum(-1).sum() / S.BS).backward()
        opt.step()
        sched.step()
        if (t + 1) % 50 == 0:
            m.eval()
            fa = acc(m, nrm, xh[hf], yh[hf])
            ra = acc(m, nrm, xh[~hf], yh[~hf])
            trace.append((t + 1, fa, ra))
            if fa <= 0.1 * nat_forget and ra >= nat_retain - 0.05:
                return t + 1, trace
    return None, trace


def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    xf, yf = S.cache("id_fit")
    xh, yh = S.cache("id_hold")
    x, y = S.cache("fresh")
    isf = torch.isin(y, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    res = dict(model=model, arm=arm, n_forget=int(isf.sum()), n_retain=int((~isf).sum()), recipe=dict(max_steps=MAXS, relearn=RSTEPS))
    ratio, dmg = locate(nat, nrm, xf[:2000], yf[:2000])
    loc = sorted(sorted(range(12), key=lambda k: -ratio[k])[:2])
    locd = sorted(sorted(range(12), key=lambda k: -dmg[k])[:2])
    gr = torch.Generator().manual_seed(95)
    rnd = sorted(torch.randperm(12, generator=gr)[:2].tolist())
    cfe = sorted(json.load(open(f"out/p18/p23_{model}_cfeG_lora1_fix_main.json"))["order"][:2])
    blocks = cfe if arm == "Rcfe" else rnd if arm == "Rrand" else loc
    res.update(ratio=ratio, damage=dmg, located=loc, located_dmg=locd, random=rnd, cfe=cfe, blocks=blocks)
    res["native"] = dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR),
                         forget_hold=acc(nat, nrm, xh[torch.isin(yh, FORGET)], yh[torch.isin(yh, FORGET)]),
                         retain_hold=acc(nat, nrm, xh[~torch.isin(yh, FORGET)], yh[~torch.isin(yh, FORGET)]))
    m = copy.deepcopy(nat)
    with torch.no_grad():
        if arm in SIG:
            gn = torch.Generator(device="cpu").manual_seed(100)
            for k in blocks:
                for q in m.blocks[k].parameters():
                    q.add_(torch.randn(q.shape, generator=gn).to(q) * SIG[arm] * q.std().clamp_min(1e-12))
        if arm == "L":
            i, j = blocks
            bi, bj = m.blocks[i], m.blocks[j]
            m.blocks[i], m.blocks[j] = bj, bi
        if arm in REM:
            m.blocks = torch.nn.Sequential(*[b for t, b in enumerate(m.blocks) if t not in blocks])
    kept = [t for t in range(12) if not (arm in REM and t in blocks)]
    m = p18.frozen(m)
    res["disrupted"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    params = [q for k in blocks for q in m.blocks[k].parameters()] if arm in ("E", "EN025") else list(m.parameters())
    steps, trace = unlearn(m, nat, nrm, xf, params, xh, yh, res["native"]["forget_hold"], res["native"]["retain_hold"])
    m = p18.frozen(m)
    res.update(steps=steps, trace=trace, reached=steps is not None)
    res["unlearned"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    print(model, arm, "blocks", blocks, "steps", steps, "native", res["native"], "unlearned", res["unlearned"], flush=True)
    res["kept"] = kept
    res["stitch"] = {k: dict(forget=stitch(m, nat, nrm, xF, yF, k, kept), retain=stitch(m, nat, nrm, xR, yR, k, kept))
                     for k in range(13)}
    rl = copy.deepcopy(m)
    xr = xf[~torch.isin(yf, FORGET)]
    ps = []
    for blk in rl.blocks:
        ps += p18.lora_wrap(blk, 4)
    p18.kd_train(rl, nat, nrm, xr, ps, RSTEPS, 1e-3, seed=0)
    rl = p18.frozen(rl)
    res["relearn"] = dict(forget=acc(rl, nrm, xF, yF), retain=acc(rl, nrm, xR, yR))
    del rl
    print(model, arm, "relearn", res["relearn"], flush=True)
    res["frame"] = {}
    for k in kept:
        r = {}
        for name, xs in (("forget", xF), ("retain", xR[:500])):
            G, _ = slot_effects(nat, nrm, xs, k, extra=[m.blocks[kept.index(k)]])
            w, nu = frame(G)
            r[name] = dict(nu=float(nu), E=float(G[12, 12] / len(xs)), change=float((G[12, 12] - 2 * G[12, k] + G[k, k]) / len(xs)))
        res["frame"][k] = r
    json.dump(res, open(f"{OUT}/p96_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
