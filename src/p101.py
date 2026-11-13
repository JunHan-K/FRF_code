"""PREREG101 (GPU): unlearning by erasing the dog-discriminative directions that blocks write, and repairing with a rank-4 LoRA that
may (unconstrained) or may not (constrained, 'C') write in those directions. Forget set, locating, repair data (no dog image, dog
logits masked), attacks and evaluation as PREREG99/100.
Directions: for every block k, image features = mean over tokens of the residual stream after block k (native network) on the dog
id_fit images and 2,000 retain id_fit images; V_k = top-r right singular vectors of (dog class means - retain mean), 118 rows.
Erasure on the selected blocks (top k by the FRF class-conditional ratio; k = 12: all): LayerScale (if any) folded into attn.proj and
mlp.fc2, then W <- (I - P_k) W, b <- (I - P_k) b with P_k = V_k V_k^T, so the block writes nothing along V_k.
Constrained repair: after every optimiser step, B <- (I - P_k) B for the LoRA of attn.proj and mlp.fc2 of the erased blocks.
Arms: P<r>-<k>[C] for r in {8, 32}, k in {4, 12}, with and without C; Prand32-12C: random orthonormal 32-dim subspaces (seed 101).
Written 2026-11-07 18:51 KST, before any PREREG101 network exists.
usage: p101.py <model> <arm>"""
import copy
import json
import math
import os
import re
import sys

import torch  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p90 import slot_effects, frame  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import Masked, locate, stitch, FORGET  # noqa: E402
from p99 import ce_attack, wrap, fold  # noqa: E402

OUT = os.environ.get("P101_OUT", "out/p101")
STEPS = int(os.environ.get("P101_STEPS", 3000))  # P101_STEPS / P101_ATT / P101_OUT: smoke test only
ATT = int(os.environ.get("P101_ATT", 300))


@torch.no_grad()
def block_features(nat, nrm, x, bs=50):
    out = []
    for s in range(0, len(x), bs):
        h = e13.emb(nat, nrm(x[s:s + bs].to(S.dev).float().div_(255)))
        f = []
        for b in nat.blocks:
            h = b(h)
            f.append(h.mean(1).double().cpu())
        out.append(torch.stack(f, 1))
    return torch.cat(out)  # N x 12 x C


def directions(fd, yd, fr, r):
    V = []
    mu_r = fr.mean(0)
    for k in range(fd.shape[1]):
        M = torch.stack([fd[yd == c, k].mean(0) for c in yd.unique()]) - mu_r[k]
        _, _, Vt = torch.linalg.svd(M, full_matrices=False)
        V.append(Vt[:r].T.float())
    return V


@torch.no_grad()
def erase(blk, P):
    for ls, lin in ((getattr(blk, "ls1", None), blk.attn.proj), (getattr(blk, "ls2", None), blk.mlp.fc2)):
        if ls is not None and hasattr(ls, "gamma"):
            lin.weight.mul_(ls.gamma[:, None])
            if lin.bias is not None:
                lin.bias.mul_(ls.gamma)
            ls.gamma.fill_(1.0)
        Q = torch.eye(P.shape[0], device=P.device) - P
        lin.weight.copy_(Q @ lin.weight)
        if lin.bias is not None:
            lin.bias.copy_(Q @ lin.bias)


def repair(m, nat, nrm, xf, params, steps, cons):
    """p18.kd_train with the dog logits masked; cons = [(lora module, Q)] projected after every step"""
    for q in m.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    teacher = Masked(nat)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(0)
    m.train()
    for t in range(steps):
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
        with torch.no_grad():
            for mod, Q in cons:
                mod.B.copy_(Q @ mod.B)
    m.eval()


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
    rank = sorted(range(12), key=lambda k: -ratio[k])
    res.update(ratio=ratio, rank=rank)
    res["native"] = dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR),
                         forget_hold=acc(nat, nrm, xh[hf], yh[hf]), retain_hold=acc(nat, nrm, xh[~hf], yh[~hf]))
    q = re.fullmatch(r"P(rand)?(\d+)-(\d+)(C?)", arm)
    rnd, r, k, cons = bool(q[1]), int(q[2]), int(q[3]), q[4] == "C"
    blocks = rank[:k]
    if rnd:
        g = torch.Generator().manual_seed(101)
        C = nat.blocks[0].mlp.fc2.out_features
        V = [torch.linalg.qr(torch.randn(C, r, generator=g))[0] for _ in range(12)]
    else:
        fd = block_features(nat, nrm, xd)
        fr = block_features(nat, nrm, xr[:2000])
        V = directions(fd, yd, fr, r)
    P = {b: (V[b] @ V[b].T).to(S.dev) for b in blocks}
    res["blocks"] = blocks
    m = copy.deepcopy(nat)
    for b in blocks:
        erase(m.blocks[b], P[b])
    m = p18.frozen(m)
    res["damaged"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    ps = wrap(m)
    cl = []
    if cons:
        for b in blocks:
            Q = torch.eye(P[b].shape[0], device=S.dev) - P[b]
            cl += [(m.blocks[b].attn.proj, Q), (m.blocks[b].mlp.fc2, Q)]
    repair(m, nat, nrm, xr, ps, STEPS, cl)
    fold(m)
    m = p18.frozen(m)
    with torch.no_grad():  # constrained arms: erased directions stay empty after the repair
        res["write_in_V"] = {b: float(max((P[b] @ m.blocks[b].attn.proj.weight).abs().max(), (P[b] @ m.blocks[b].mlp.fc2.weight).abs().max()))
                             for b in blocks}
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
        rr = {}
        for name, xs in (("forget", xF), ("retain", xR[:500])):
            G, _ = slot_effects(nat, nrm, xs, t, extra=[m.blocks[t]])
            w, nu = frame(G)
            rr[name] = dict(nu=float(nu), E=float(G[12, 12] / len(xs)))
        res["frame"][t] = rr
    json.dump(res, open(f"{OUT}/p101_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
