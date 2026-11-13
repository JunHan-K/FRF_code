"""PREREG62: the E3 size controls (PREREG50) re-measured in the preregistered primary context of PREREG57.
Context pri: native input entering slot k, native suffix after k. Metric: common Fisher (p = the native network's own
prediction, native block k in slot k). Per cell, start block s = KEEP[k] (s = k for in-place cells) and candidates
  base      : the recovered block B_u
  copy_s    : B_s (known answer nu = 0; B_s is one of the references)
  scale_loc : h + g_loc (B_s(h) - h), g_loc = sum_n |B_u(h_n) - h_n| / sum_n |B_s(h_n) - h_n| on calib[:500] (native input)
  scale_en  : g_en by bisection on log g in [1/64, 64] (24 steps) so that the common-Fisher energy of the output effect on
              calib[:500] equals that of B_u
Stored per image: Gram of (12 native references, base, copy_s, scale_loc, scale_en) in the common Fisher, for calib =
id_fit[:1000] (float64) and fresh (float32, only with P62_FRESH=1).
One job = one network, all its cells; existing cell files are skipped.
usage: p62.py <model> <L|E> <perm> [comma-separated slots]"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p9lib as P  # noqa: E402
import p11  # noqa: E402
import s12r  # noqa: E402
import spec  # noqa: E402
from p47 import e_paths  # noqa: E402

OUT = os.environ.get("P62_OUT", "out/p62")
MB, NG = 25, 500


@torch.no_grad()
def main(model, kind, p, slots=None):
    if kind == "L":
        run, pt, _ = p11.load_case(model, p)
    else:
        D.M = s12r.MODELS[model]
        run, pt, _ = e_paths(model, p, kind)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    for q in list(nat.parameters()) + list(m.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(m.blocks)
    ks = [k for k in range(6, L) if (D.KEEP[run][k] != k if kind == "L" else D.KEEP[run][k] == k)]
    if slots:
        ks = [int(t) for t in slots.split(",")]
    sets = [("calib", "id_fit", 1000)] + ([("fresh", "fresh", None)] if os.environ.get("P62_FRESH") else [])
    os.makedirs(OUT, exist_ok=True)

    def prefix(x, k):  # native input to slot k
        h = e13.emb(nat, x)
        for b in nat.blocks[:k]:
            h = b(h)
        return h

    def suf(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).double()

    xc, _ = S.cache("id_fit")
    for k in ks:
        fn = f"{OUT}/p62_{model}_{kind}_{p}_{k}.npz"
        if os.path.exists(fn):
            continue
        s_ = D.KEEP[run][k]
        Bs, Bu, Bk = nat.blocks[s_], m.blocks[k], nat.blocks[k]
        H = [prefix(nrm(xc[s:s + MB].to(S.dev).float().div_(255)), k) for s in range(0, NG, MB)]
        ZS = [suf(h, k) for h in H]
        PC = [suf(Bk(h), k).softmax(-1)[:, None] for h in H]

        def energy(f):
            tot = 0.0
            for h, zs, pc in zip(H, ZS, PC):
                e = (suf(f(h), k) - zs)[:, None]
                tot += float(spec.fip(e, e, pc)[:, 0, 0].sum())
            return tot / NG

        g_loc = sum(float((Bu(h) - h).flatten(1).double().norm(dim=1).sum()) for h in H) / \
            max(sum(float((Bs(h) - h).flatten(1).double().norm(dim=1).sum()) for h in H), 1e-30)
        e_base = energy(lambda h: Bu(h))
        lo, hi = math.log(1 / 64), math.log(64)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if energy(lambda h, g=math.exp(mid): h + g * (Bs(h) - h)) < e_base:
                lo = mid
            else:
                hi = mid
        g_en = math.exp(0.5 * (lo + hi))
        e_en = energy(lambda h: h + g_en * (Bs(h) - h))
        C = [lambda h: Bu(h), lambda h: Bs(h), lambda h: h + g_loc * (Bs(h) - h), lambda h: h + g_en * (Bs(h) - h)]
        res = dict(model=model, kind=kind, perm=p, run=run, slot=k, src=s_, g_loc=g_loc, g_en=g_en, e_base=e_base, e_en=e_en,
                   cands=["base", "copy_s", "scale_loc", "scale_en"])
        out = {}
        for sname, cname, n in sets:
            x, y = S.cache(cname)
            x, y = (x[:n], y[:n]) if n else (x, y)
            G = []
            for s in range(0, len(x), MB):
                h = prefix(nrm(x[s:s + MB].to(S.dev).float().div_(255)), k)
                zs = suf(h, k)
                Z = torch.stack([suf(nat.blocks[b](h), k) for b in range(L)], 1)
                Zc = torch.stack([suf(f(h), k) for f in C], 1)
                A = torch.cat([Z, Zc], 1) - zs[:, None]
                G.append(spec.fip(A, A, Z[:, k].softmax(-1)[:, None]).cpu())
            G = torch.cat(G).numpy()
            out[f"{sname}/G"] = G if sname == "calib" else G.astype(np.float32)
            out[f"{sname}/label"] = y.numpy()
            print(model, kind, p, k, sname, len(x), round(g_loc, 3), round(g_en, 3), flush=True)
        np.savez(fn, meta=np.array(repr(res)), **out)


if __name__ == "__main__":
    main(*sys.argv[1:])
