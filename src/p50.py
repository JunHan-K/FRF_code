"""PREREG50 (E3): is the out-of-frame share of a retrained block more than what a pure change of update size would give?
Even a native block whose update is only rescaled (h + g (B(h) - h)) leaves the span of the 12 native effects through the
nonlinear suffix (p42: scale_2.0 median nu_span12 0.13). Per cell, the start block s = KEEP[k] (the native block the slot
held before retraining; s = k for in-place cells) is rescaled to the size of the retrained block B_u:
  scale_loc : g_loc = sum_n |B_u(h_n) - h_n| / sum_n |B_s(h_n) - h_n| on id_fit[:500] (local update norm, closed form)
  scale_en  : g_en by bisection on log g in [1/64, 64] (24 steps) so that the Fisher energy of the output effect
              (metric p_candidate, origin skip, mean over id_fit[:500]) equals that of B_u
  copy_s    : g = 1 (known answer nu = 0)
Measurement identical to p47 (nu_span12, native suffix after k, spec.fit), so base reproduces p47.
usage: p50.py <model> <L|E> <perm> [comma-separated slots]"""
import json
import math
import os
import sys

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

OUT = "out/p50"
NA, MB = 500, int(os.environ.get("P50_MB", 25))


def rownorm(t):
    return t.flatten(1).double().norm(dim=1)


@torch.no_grad()
def main(model, kind, p, slots=None):
    os.makedirs(OUT, exist_ok=True)
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
    ks = [int(t) for t in slots.split(",")] if slots else \
        [k for k in range(6, L) if (D.KEEP[run][k] != k if kind == "L" else D.KEEP[run][k] == k)]
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])

    def prefix(x, k):
        h = e13.emb(m, x)
        for b in m.blocks[:k]:
            h = b(h)
        return h

    def suf_nat(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, kind=kind, perm=p, run=run, n_fit=NA, cells={})
    for k in ks:
        s_ = D.KEEP[run][k]
        Bs, Bu = nat.blocks[s_], m.blocks[k]
        H = [prefix(xf[s:s + MB].to(S.dev), k).cpu() for s in range(0, NA, MB)]
        ZS = [suf_nat(h.to(S.dev), k).cpu() for h in H]

        def energy(f):
            tot = 0.0
            for h, zs in zip(H, ZS):
                h, zs = h.to(S.dev), zs.to(S.dev)
                z = suf_nat(f(h), k)
                e = (z - zs)[:, None]
                tot += float(spec.fip(e, e, z.softmax(-1)[:, None])[:, 0, 0].sum())
            return tot / NA

        nu_ = sum(float(rownorm(Bu(h.to(S.dev)) - h.to(S.dev)).sum()) for h in H)
        ns_ = sum(float(rownorm(Bs(h.to(S.dev)) - h.to(S.dev)).sum()) for h in H)
        g_loc = nu_ / max(ns_, 1e-30)
        e_base = energy(lambda h: Bu(h))
        lo, hi = math.log(1 / 64), math.log(64)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if energy(lambda h, g=math.exp(mid): h + g * (Bs(h) - h)) < e_base:
                lo = mid
            else:
                hi = mid
        g_en = math.exp(0.5 * (lo + hi))
        C = {"base": lambda h: Bu(h), "copy_s": lambda h: Bs(h),
             "scale_loc": lambda h: h + g_loc * (Bs(h) - h), "scale_en": lambda h: h + g_en * (Bs(h) - h)}
        acc = {nm: dict(G=torch.zeros(L, L, dtype=torch.float64), c=torch.zeros(L, dtype=torch.float64), u=0.0) for nm in C}
        for h, zs in zip(H, ZS):
            h, zs = h.to(S.dev), zs.to(S.dev)
            E = torch.stack([suf_nat(nat.blocks[b](h), k) - zs for b in range(L)], 1)
            for nm, f in C.items():
                z = suf_nat(f(h), k)
                a = acc[nm]
                pz = z.softmax(-1)[:, None]
                e = (z - zs)[:, None]
                a["G"] += spec.fip(E, E, pz).sum(0).cpu()
                a["c"] += spec.fip(E, e, pz)[..., 0].sum(0).cpu()
                a["u"] += float(spec.fip(e, e, pz)[:, 0, 0].sum())
        cell = dict(s=s_, k=k, g_loc=g_loc, g_en=g_en, cands={})
        for nm, a in acc.items():
            Fit = spec.fit(a["G"], a["c"], torch.tensor(a["u"], dtype=torch.float64))
            cell["cands"][nm] = dict(nu_span12=float(Fit["nu_span"]), energy=a["u"] / NA)
        res["cells"][k] = cell
        print(model, kind, p, k, s_, round(g_loc, 3), round(g_en, 3),
              {nm: round(v["nu_span12"], 4) for nm, v in cell["cands"].items()}, flush=True)
    tag = f"_{slots.replace(',', '-')}" if slots else ""
    json.dump(res, open(f"{OUT}/p50_{model}_{kind}_{p}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
