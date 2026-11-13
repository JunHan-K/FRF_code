"""PREREG99 (GPU): unlearning = damage the blocks that carry a class group, then repair ONLY the damage with a small LoRA on the
retained classes. Forget set F = the 118 ImageNet dog classes (151-268). No dog image is used in recovery, and the teacher's dog
logits are masked, so nothing about dogs is given back.
Locate on id_fit[:2000]: blocks ranked by the class-conditional FRF magnitude ratio (energy of the block's effect on forget images
/ on retain images).
Arms (k = number of top-ranked blocks):
  U0 no damage; HR no damage, no recovery, dog rows of the head re-initialised (output-only erasure reference);
  N<s>-<k> Gaussian noise s x tensor std on the k located blocks; L-<k> located blocks swap slots in rank pairs; LN-<k> both, s = 0.5;
  R-<k> located blocks removed; N1-rand4 noise 1.0 on 4 random blocks (seed 99).
Recovery: rank-4 LoRA on every block, distillation from the native network with dog logits masked, retain-class id_fit images
only, 3,000 steps, lr 1e-3, seed 0 (no early stop).
Measured: forget / retain top-1 on id_hold (selection) and fresh (evaluation); stitch(k); frame of each block; attacks on the
repaired network, each from a copy, rank-4 LoRA: (a) hard labels, retain images only, 300 steps; (b) hard labels, one dog image per
class from id_fit, 100 steps; (c) hard labels, all dog images of id_fit, 300 steps; (d) distillation from the unmasked native
network on retain images, 300 steps (as PREREG91-96). (a) cannot bring dogs back by construction (it pushes dog logits down) and
is descriptive; (b) and (c) are read against HR, whose features are intact.
Written 2026-11-07 12:55 KST, before any PREREG99 network exists.
usage: p99.py <model> <arm>"""
import copy
import json
import os
import re
import sys

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p11  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p90 import slot_effects, frame  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import Masked, locate, stitch, FORGET  # noqa: E402

OUT = os.environ.get("P99_OUT", "out/p99")
STEPS = int(os.environ.get("P99_STEPS", 3000))  # P99_STEPS / P99_ATT / P99_OUT: smoke test only
ATT = int(os.environ.get("P99_ATT", 300))
RANK = 4


def fold(m):
    for blk in m.blocks:
        for parent, nme in p11.lora_targets(blk):
            mod = getattr(parent, nme)
            with torch.no_grad():
                mod.lin.weight += mod.B @ mod.A
            setattr(parent, nme, mod.lin)


def wrap(m):
    ps = []
    for blk in m.blocks:
        ps += p18.lora_wrap(blk, RANK)
    return ps


def ce_attack(m, nrm, x, y, steps):
    """hard-label fine-tuning with a rank-4 LoRA from a copy of m; returns the attacked network"""
    a = copy.deepcopy(m)
    ps = wrap(a)
    for q in a.parameters():
        q.requires_grad_(False)
    for q in ps:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    a.train()
    for _ in range(steps):
        idx = torch.randint(len(x), (S.BS,), generator=g)
        xb = nrm(S.augment(x[idx].to(S.dev, non_blocking=True), g))
        yb = y[idx].to(S.dev)
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            (nn.functional.cross_entropy(a(xb[s:s + S.MB]).float(), yb[s:s + S.MB], reduction="sum") / S.BS).backward()
        opt.step()
    return p18.frozen(a)


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
    res = dict(model=model, arm=arm, n_forget=int(isf.sum()), recipe=dict(rank=RANK, steps=STEPS, attack=ATT))
    ratio, _ = locate(nat, nrm, xf[:2000], yf[:2000])
    rank = sorted(range(12), key=lambda k: -ratio[k])
    res.update(ratio=ratio, rank=rank)
    res["native"] = dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR),
                         forget_hold=acc(nat, nrm, xh[hf], yh[hf]), retain_hold=acc(nat, nrm, xh[~hf], yh[~hf]))
    q = re.fullmatch(r"(N|L|LN|R)([\d.]*)-(\d)", arm)
    kind, sig, k = (q[1], float(q[2] or 0.5), int(q[3])) if q else (None, 1.0, 4)
    if arm == "N1-rand4":
        blocks = sorted(torch.randperm(12, generator=torch.Generator().manual_seed(99))[:4].tolist())
        kind = "N"
    else:
        blocks = rank[:k] if kind else []
    res["blocks"] = blocks
    m = copy.deepcopy(nat)
    with torch.no_grad():
        if kind in ("N", "LN"):
            gn = torch.Generator().manual_seed(100)
            for b in blocks:
                for p in m.blocks[b].parameters():
                    p.add_(torch.randn(p.shape, generator=gn).to(p) * sig * p.std().clamp_min(1e-12))
        order = list(range(12))
        if kind in ("L", "LN"):
            for a_, b_ in zip(blocks[0::2], blocks[1::2]):
                order[a_], order[b_] = order[b_], order[a_]
        kept = [t for t in range(12) if not (kind == "R" and t in blocks)]
        m.blocks = nn.Sequential(*[m.blocks[order[t]] for t in kept])
        if arm == "HR":
            w = m.head.weight
            w[FORGET.to(w.device)] = torch.randn(len(FORGET), w.shape[1], generator=torch.Generator().manual_seed(99)).to(w) * w.std()
            m.head.bias[FORGET.to(w.device)] = 0
    m = p18.frozen(m)
    res["kept"] = kept
    res["damaged"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    xr, yr = xf[~ff], yf[~ff]
    if arm != "HR":
        ps = wrap(m)
        p18.kd_train(m, Masked(nat), nrm, xr, ps, STEPS, 1e-3, seed=0)
        fold(m)
        m = p18.frozen(m)
    res["repaired"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR),
                           forget_hold=acc(m, nrm, xh[hf], yh[hf]), retain_hold=acc(m, nrm, xh[~hf], yh[~hf]))
    print(model, arm, "blocks", blocks, "damaged", res["damaged"], "repaired", res["repaired"], flush=True)
    res["stitch"] = {t: stitch(m, nat, nrm, xF, yF, t, kept) for t in range(13)}
    a = ce_attack(m, nrm, xr, yr, ATT)
    res["attack_hard_retain"] = dict(forget=acc(a, nrm, xF, yF), retain=acc(a, nrm, xR, yR))
    del a
    idx = torch.tensor([int(torch.nonzero(yf == c)[0]) for c in FORGET.tolist() if (yf == c).any()])
    a = ce_attack(m, nrm, xf[idx], yf[idx], ATT // 3)
    res["attack_fewshot"] = dict(forget=acc(a, nrm, xF, yF), retain=acc(a, nrm, xR, yR), n=len(idx))
    del a
    a = ce_attack(m, nrm, xf[ff], yf[ff], ATT)
    res["attack_fullshot"] = dict(forget=acc(a, nrm, xF, yF), retain=acc(a, nrm, xR, yR), n=int(ff.sum()))
    del a
    a = copy.deepcopy(m)
    p18.kd_train(a, nat, nrm, xr, wrap(a), ATT, 1e-3, seed=0)
    a = p18.frozen(a)
    res["attack_teacher"] = dict(forget=acc(a, nrm, xF, yF), retain=acc(a, nrm, xR, yR))
    del a
    print(model, arm, {t: res[t] for t in ("attack_hard_retain", "attack_fewshot", "attack_fullshot", "attack_teacher")}, flush=True)
    res["frame"] = {}
    for t in kept:
        r = {}
        for name, xs in (("forget", xF), ("retain", xR[:500])):
            G, _ = slot_effects(nat, nrm, xs, t, extra=[m.blocks[kept.index(t)]])
            w, nu = frame(G)
            r[name] = dict(nu=float(nu), E=float(G[12, 12] / len(xs)))
        res["frame"][t] = r
    json.dump(res, open(f"{OUT}/p99_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
