"""PREREG100 (GPU): unlearning by TARGETED damage and small repair. Same forget set, locating rule, repair, attacks and evaluation as
PREREG99 (p99.py); only the damage differs, and it is aimed at the dog group (dog images are used for the damage, never for the
repair). Located blocks = top k by the FRF class-conditional magnitude ratio (k = 12: every block).
  GA-<k>   gradient ascent on the dog cross-entropy, parameters of the k blocks, AdamW lr 1e-4, no decay, batch 64 dog id_fit images;
           stop at the first 10-step check with dog top-1 on those images <= 0.1 x native, at most 500 steps.
  GAR-<k>  the same with batch 32 dog + 32 retain images, adding distillation to the native network on the retain part.
  SD<a>-<k> selective dampening: diagonal Fisher of the k blocks on 400 dog and 800 retain id_fit images (true labels, one image at a
           time); where I_dog > a x I_retain, the parameter is multiplied by min(I_retain / I_dog, 1). a = 10 or 3.
  GA-rand4 GA on 4 random blocks (seed 100).
Written 2026-11-07 14:39 KST, before any PREREG100 network exists.
usage: p100.py <model> <arm>"""
import copy
import json
import os
import re
import sys

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p90 import slot_effects, frame  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import Masked, locate, stitch, FORGET  # noqa: E402
from p99 import ce_attack, wrap, fold  # noqa: E402

OUT = os.environ.get("P100_OUT", "out/p100")
STEPS = int(os.environ.get("P100_STEPS", 3000))  # P100_STEPS / P100_ATT / P100_OUT: smoke test only
ATT = int(os.environ.get("P100_ATT", 300))
MAXGA = 500


def ascent(m, nat, nrm, params, xd, yd, xr, guard):
    for q in m.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    nd = S.BS // 2 if guard else S.BS
    target = 0.1 * acc(nat, nrm, xd[:300], yd[:300])
    for t in range(1, MAXGA + 1):
        m.train()
        idx = torch.randint(len(xd), (nd,), generator=g)
        xb, yb = nrm(S.augment(xd[idx].to(S.dev), g)), yd[idx].to(S.dev)
        opt.zero_grad(set_to_none=True)
        for s in range(0, nd, S.MB):
            (-F.cross_entropy(m(xb[s:s + S.MB]).float(), yb[s:s + S.MB], reduction="sum") / S.BS).backward()
        if guard:
            jdx = torch.randint(len(xr), (S.BS - nd,), generator=g)
            xc = nrm(S.augment(xr[jdx].to(S.dev), g))
            for s in range(0, S.BS - nd, S.MB):
                lps = m(xc[s:s + S.MB]).log_softmax(-1)
                with torch.no_grad():
                    lpt = nat(xc[s:s + S.MB]).log_softmax(-1)
                ((lpt.exp() * (lpt - lps)).sum(-1).sum() / S.BS).backward()
        opt.step()
        if t % 10 == 0:
            m.eval()
            if acc(m, nrm, xd[:300], yd[:300]) <= target:
                return t
    return None


def fisher(m, nrm, params, x, y):
    out = [torch.zeros_like(q) for q in params]
    for q in m.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    m.eval()
    for i in range(len(x)):
        z = m(nrm(x[i:i + 1].to(S.dev).float().div_(255))).float()
        gs = torch.autograd.grad(F.cross_entropy(z, y[i:i + 1].to(S.dev)), params)
        for o, g_ in zip(out, gs):
            o += g_.detach() ** 2
    return [o / len(x) for o in out]


@torch.no_grad()
def dampen(params, If, Ir, a):
    frac = []
    for q, f_, r_ in zip(params, If, Ir):
        mask = f_ > a * r_
        beta = (r_ / f_.clamp_min(1e-30)).clamp(max=1.0)
        q[mask] *= beta[mask]
        frac.append(float(mask.float().mean()))
    return sum(frac) / len(frac)


def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    xf, yf = S.cache("id_fit")
    xh, yh = S.cache("id_hold")
    x, y = S.cache("fresh")
    isf, hf, ff = torch.isin(y, FORGET), torch.isin(yh, FORGET), torch.isin(yf, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    xd, yd, xr, yr = xf[ff], yf[ff], xf[~ff], yf[~ff]
    res = dict(model=model, arm=arm, n_forget=int(isf.sum()), n_damage_images=int(ff.sum()), recipe=dict(steps=STEPS, attack=ATT))
    ratio, _ = locate(nat, nrm, xf[:2000], yf[:2000])
    rank = sorted(range(12), key=lambda k: -ratio[k])
    res.update(ratio=ratio, rank=rank)
    res["native"] = dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR),
                         forget_hold=acc(nat, nrm, xh[hf], yh[hf]), retain_hold=acc(nat, nrm, xh[~hf], yh[~hf]))
    if arm == "GA-rand4":
        kind, a, blocks = "GA", None, sorted(torch.randperm(12, generator=torch.Generator().manual_seed(100))[:4].tolist())
    else:
        q = re.fullmatch(r"(GAR|GA|SD)(\d*)-(\d+)", arm)
        kind, a, blocks = q[1], float(q[2]) if q[2] else None, rank[:int(q[3])]
    res["blocks"] = blocks
    m = copy.deepcopy(nat)
    params = [p for b in blocks for p in m.blocks[b].parameters()]
    if kind in ("GA", "GAR"):
        res["ascent_steps"] = ascent(m, nat, nrm, params, xd, yd, xr, kind == "GAR")
    else:
        If = fisher(m, nrm, params, xd[:400], yd[:400])
        Ir = fisher(m, nrm, params, xr[:800], yr[:800])
        res["dampened_fraction"] = dampen(params, If, Ir, a)
    m = p18.frozen(m)
    res["damaged"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    ps = wrap(m)
    p18.kd_train(m, Masked(nat), nrm, xr, ps, STEPS, 1e-3, seed=0)
    fold(m)
    m = p18.frozen(m)
    res["repaired"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR),
                           forget_hold=acc(m, nrm, xh[hf], yh[hf]), retain_hold=acc(m, nrm, xh[~hf], yh[~hf]))
    print(model, arm, "blocks", blocks, "damaged", res["damaged"], "repaired", res["repaired"], flush=True)
    kept = list(range(12))
    res["kept"] = kept
    res["stitch"] = {t: stitch(m, nat, nrm, xF, yF, t, kept) for t in range(13)}
    a_ = ce_attack(m, nrm, xr, yr, ATT)
    res["attack_hard_retain"] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR))
    del a_
    idx = torch.tensor([int(torch.nonzero(yf == c)[0]) for c in FORGET.tolist() if (yf == c).any()])
    a_ = ce_attack(m, nrm, xf[idx], yf[idx], ATT // 3)
    res["attack_fewshot"] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR), n=len(idx))
    del a_
    a_ = ce_attack(m, nrm, xd, yd, ATT)
    res["attack_fullshot"] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR), n=int(ff.sum()))
    del a_
    a_ = copy.deepcopy(m)
    p18.kd_train(a_, nat, nrm, xr, wrap(a_), ATT, 1e-3, seed=0)
    a_ = p18.frozen(a_)
    res["attack_teacher"] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR))
    del a_
    print(model, arm, {t: res[t] for t in ("attack_fewshot", "attack_fullshot", "attack_teacher")}, flush=True)
    res["frame"] = {}
    for t in kept:
        r = {}
        for name, xs in (("forget", xF), ("retain", xR[:500])):
            G, _ = slot_effects(nat, nrm, xs, t, extra=[m.blocks[t]])
            w, nu = frame(G)
            r[name] = dict(nu=float(nu), E=float(G[12, 12] / len(xs)))
        res["frame"][t] = r
    json.dump(res, open(f"{OUT}/p100_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
