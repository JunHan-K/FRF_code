"""PREREG67 Q6: independent recomputation of stored endpoints for a hashed 10% of new-parent cells.
(1) nu and energy: explicit per-image Fisher matrices F = diag(p) - p p' (float64, CPU), effects of the 12 native blocks
    and the candidate read through the native suffix on native input (p = native network's prediction), normal equations
    solved with numpy lstsq (no eigenvalue cut-off), on calib = id_fit[:200]. Compared with the stored Gram path
    (p57 calib/pri/G_common, first 200 images, pinv with the primary tau).
(2) replacement: the recovered network run block by block (embedding, blocks with slot k replaced by native block b, norm,
    head) on fresh[:300], top-1 compared with the stored p59 pred_rep[:300, b] for one hashed b.
usage: p69.py <model> <L|E|N> <perm> <slot>"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p60 import parent_net  # noqa: E402

OUT = "out/p69"
P57 = ("out/p65", "out/p66par")
P59 = ("out/p65rep", "out/p66rep")
dev = S.dev
tau = 1e-8  # relative eigenvalue cut-off of the pseudo-inverse (the primary tau of the FRF measurement)


@torch.no_grad()
def main(model, kind, perm, slot):
    k = int(slot)
    D.M = s12r.MODELS[model]
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    run, m = parent_net(model, kind, perm)
    L = len(m.blocks)
    xc, _ = S.cache("id_fit")
    G = np.zeros((L + 1, L + 1))
    for s in range(0, 200, 20):
        x = nrm(xc[s:s + 20].to(dev).float().div_(255))
        h = nat.norm_pre(nat._pos_embed(nat.patch_embed(x)))
        for blk in nat.blocks[:k]:
            h = blk(h)

        def suf(o):
            for blk in nat.blocks[k + 1:]:
                o = blk(o)
            return nat.forward_head(nat.norm(o)).double().cpu().numpy()

        z0 = suf(h)
        Z = [suf(nat.blocks[b](h)) for b in range(L)] + [suf(m.blocks[k](h))]
        P = np.exp(Z[k] - Z[k].max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        for n in range(len(z0)):
            F = np.diag(P[n]) - np.outer(P[n], P[n])
            Eff = np.stack([z[n] - z0[n] for z in Z])  # (13, C)
            G += Eff @ F @ Eff.T
    A, bb, u = G[:L, :L], G[:L, L], G[L, L]
    w = np.linalg.lstsq(A, bb, rcond=None)[0]
    nu_ind = float((u - 2 * w @ bb + w @ A @ w) / u)
    zf = next(np.load(f"{d_}/p57_{model}_{kind}_{perm}_{k}.npz") for d_ in P57 if os.path.exists(f"{d_}/p57_{model}_{kind}_{perm}_{k}.npz"))
    Gs = zf["calib/pri/G_common"][:200].sum(0).astype(np.float64)
    A2, b2, u2 = Gs[:L, :L], Gs[:L, 12], Gs[12, 12]
    lam, V = np.linalg.eigh(A2)
    keep = lam > tau * lam.max()
    w2 = V[:, keep] @ ((V[:, keep].T @ b2) / lam[keep])
    nu_gram = float((u2 - 2 * w2 @ b2 + w2 @ A2 @ w2) / u2)
    res = dict(model=model, kind=kind, perm=perm, slot=k, nu_indep=nu_ind, nu_gram=nu_gram, energy_indep=u / 200, energy_gram=u2 / 200)
    b = int(hashlib.sha1(f"{model}|{kind}|{perm}|{k}|ref".encode()).hexdigest(), 16) % L
    xf, _ = S.cache("fresh")
    pr = []
    for s in range(0, 300, 50):
        x = nrm(xf[s:s + 50].to(dev).float().div_(255))
        h = m.norm_pre(m._pos_embed(m.patch_embed(x)))
        for t, blk in enumerate(m.blocks):
            h = (nat.blocks[b] if t == k else blk)(h)
        pr.append(m.forward_head(m.norm(h)).argmax(-1).cpu().numpy())
    pr = np.concatenate(pr)
    zr = next(np.load(f"{d_}/p59_{model}_{kind}_{perm}.npz") for d_ in P59 if os.path.exists(f"{d_}/p59_{model}_{kind}_{perm}.npz"))
    res.update(ref=b, rep_match=float((zr[f"fresh/{k}/pred_rep"][:300, b] == pr).mean()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/p69_{model}_{kind}_{perm}_{k}.json", "w"), indent=1)
    print(res, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
