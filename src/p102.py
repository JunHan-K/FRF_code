"""PREREG102 (GPU): relearning test with matched heads. Every arm re-initialises the dog head rows exactly as HR (seed 99) and then gets the same dog-free repair, so the only
difference from the reference is the block damage.
Arms (all blocks, k = 12; damage as PREREG100 / PREREG101):
  HRr       head re-init + repair (reference)
  SD3-12H, SD10-12H   selective dampening (PREREG100) + head re-init + repair
  P32-12CH, P8-12CH   dog-direction erasure (PREREG101) + head re-init + constrained repair
Repair: p101.repair (rank-4 LoRA on all blocks, masked-teacher distillation, retain id_fit images only, 3,000 steps, seed 0).
Attacks and evaluation as PREREG99-101.
Written 2026-11-08 00:30 KST, before any PREREG102 network exists.
usage: p102.py <model> <arm>"""
import copy
import json
import os
import sys

import torch  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p90 import slot_effects, frame  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import locate, stitch, FORGET  # noqa: E402
from p99 import ce_attack, wrap, fold  # noqa: E402
from p100 import fisher, dampen  # noqa: E402
from p101 import block_features, directions, erase, repair  # noqa: E402

OUT = os.environ.get("P102_OUT", "out/p102")
STEPS = int(os.environ.get("P102_STEPS", 3000))  # P102_STEPS / P102_ATT / P102_OUT: smoke test only
ATT = int(os.environ.get("P102_ATT", 300))


@torch.no_grad()
def head_reinit(m):
    w = m.head.weight
    w[FORGET.to(w.device)] = torch.randn(len(FORGET), w.shape[1], generator=torch.Generator().manual_seed(99)).to(w) * w.std()
    m.head.bias[FORGET.to(w.device)] = 0


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
    res = dict(model=model, arm=arm, n_forget=int(isf.sum()), recipe=dict(steps=STEPS, attack=ATT))
    ratio, _ = locate(nat, nrm, xf[:2000], yf[:2000])
    res.update(ratio=ratio, rank=sorted(range(12), key=lambda k: -ratio[k]))
    res["native"] = dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR),
                         forget_hold=acc(nat, nrm, xh[hf], yh[hf]), retain_hold=acc(nat, nrm, xh[~hf], yh[~hf]))
    blocks = list(range(12))
    m = copy.deepcopy(nat)
    cl = []
    if arm.startswith("SD"):
        a = float(arm[2:].split("-")[0])
        params = [p for b in blocks for p in m.blocks[b].parameters()]
        If = fisher(m, nrm, params, xd[:400], yd[:400])
        Ir = fisher(m, nrm, params, xr[:800], yr[:800])
        res["dampened_fraction"] = dampen(params, If, Ir, a)
    elif arm.startswith("P"):
        r = int(arm[1:].split("-")[0])
        V = directions(block_features(nat, nrm, xd), yd, block_features(nat, nrm, xr[:2000]), r)
        P = {b: (V[b] @ V[b].T).to(S.dev) for b in blocks}
        for b in blocks:
            erase(m.blocks[b], P[b])
    elif arm != "HRr":
        raise ValueError(arm)
    head_reinit(m)
    m = p18.frozen(m)
    res["damaged"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    ps = wrap(m)
    if arm.startswith("P"):
        for b in blocks:
            Q = torch.eye(P[b].shape[0], device=S.dev) - P[b]
            cl += [(m.blocks[b].attn.proj, Q), (m.blocks[b].mlp.fc2, Q)]
    repair(m, nat, nrm, xr, ps, STEPS, cl)
    fold(m)
    m = p18.frozen(m)
    if arm.startswith("P"):
        with torch.no_grad():
            res["write_in_V"] = {b: float(max((P[b] @ m.blocks[b].attn.proj.weight).abs().max(), (P[b] @ m.blocks[b].mlp.fc2.weight).abs().max()))
                                 for b in blocks}
    res["repaired"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR),
                           forget_hold=acc(m, nrm, xh[hf], yh[hf]), retain_hold=acc(m, nrm, xh[~hf], yh[~hf]))
    print(model, arm, "damaged", res["damaged"], "repaired", res["repaired"], flush=True)
    res["kept"] = blocks
    res["stitch"] = {t: stitch(m, nat, nrm, xF, yF, t, blocks) for t in range(13)}
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
    for t in blocks:
        rr = {}
        for name, xs in (("forget", xF), ("retain", xR[:500])):
            G, _ = slot_effects(nat, nrm, xs, t, extra=[m.blocks[t]])
            w, nu = frame(G)
            rr[name] = dict(nu=float(nu), E=float(G[12, 12] / len(xs)))
        res["frame"][t] = rr
    json.dump(res, open(f"{OUT}/p102_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
