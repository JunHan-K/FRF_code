"""PREREG57 (P0-A/B): per-image Gram statistics of block effects for exact-projection / estimator checks and for the
input x suffix x metric decomposition. One job = one recovered network and one slot.
Contexts (input entering slot k, suffix after k):  pri = (native, native)  rep = (recovered, native)
                                                    rr  = (recovered, recovered)  nr = (native, recovered)   [ViT-B: pri, rep]
Effects e = z(X) - z(skip) in the context, for the 12 native blocks (references, recomputed in every context), the
recovered block (base) and two synthetic known-answer candidates:
  lin   = 0.5 e_k + 0.3 e_{k-1} + 0.2 e_0                     (inside the span: nu = 0)
  plant = e_k + s_x r_x, r_x = g_x minus its per-image projection on the references in the metric, g_x a fixed random
          logit vector (generator seed 1000 + k), s_x so that |s_x r_x|^2 = rho/(1-rho) |e_k|^2 with rho = 0.3 (nu = 0.3)
Metrics: common Fisher (p = the context network's own prediction with its own block in slot k: native suffix -> B_k,
recovered suffix -> B_u), centred Euclidean (p uniform), candidate Fisher (p = the candidate's own prediction; Gram of the
12 references + that candidate, for base / copy_k / lin). plant is built separately inside each fixed metric.
Stored per image (float64): G_common (15x15: refs, base, lin, plant), G_euclid (15x15), G_cand (3 x 13x13); predictions
(argmax) of skip / refs / base, the label, KL(base || ref_b). CKA of block updates (native and recovered input).
Image sets: calib = id_fit[:1000] (all contexts), hold = id_hold[:500] (all contexts), shiftR = shift_R[:500] (pri),
fresh = cache/fresh.npz (pri and rep; only with P57_FRESH=1, never in smoke tests). P57_RAW=1 also stores the raw effects of
the first 100 calib images in context pri (float64) for the direct-SVD check.
usage: p57.py <model> <L|E|N> <perm> <slot> [contexts, default by model size]"""
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

OUT = os.environ.get("P57_OUT", "out/p57")
MB, RHO = 25, 0.3


def lin_cka(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    return float((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm()).clamp_min(1e-30))


def feat(d):
    return torch.cat([d[:, 0], d[:, 1:].mean(1)], 1).double().cpu()


def plant(E, k, p, gen):
    """per-image known out-of-frame candidate in the metric p: e_k + s r, r orthogonal (per image) to all references"""
    n, L, C = E.shape
    g = torch.randn(n, 1, C, generator=gen).to(E)
    G = spec.fip(E, E, p)
    b = spec.fip(E, g, p)[..., 0]
    w = torch.linalg.pinv(G, rtol=1e-12) @ b[..., None]
    r = g - (w.to(E.dtype).transpose(1, 2) @ E)
    rr = spec.fip(r, r, p)[:, 0, 0]
    ek = E[:, k:k + 1]
    ee = spec.fip(ek, ek, p)[:, 0, 0]
    s = (RHO / (1 - RHO) * ee / rr.clamp_min(1e-30)).sqrt()
    return ek + s.to(E.dtype)[:, None, None] * r


@torch.no_grad()
def main(model, kind, p, slot, ctxs=None):
    k = int(slot)
    if kind == "L":
        run, pt, _ = p11.load_case(model, p)
    else:
        D.M = s12r.MODELS[model]
        run, pt, _ = e_paths(model, p, kind)
    big = model.endswith("_b")
    ctxs = (ctxs or ("pri,rep" if big else "pri,rep,rr,nr")).split(",")
    pt = os.environ.get("P57_PT", pt)  # PREREG58: an intermediate checkpoint (measured, then deleted: latest-only policy)
    lite = os.environ.get("P57_LITE", "0").lower() not in ("", "0", "false", "no")
    if lite:
        ctxs = ["pri"]
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    for q in list(nat.parameters()) + list(m.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L, Bu = len(m.blocks), m.blocks[k]
    sets = [("calib", "id_fit", 1000, ctxs), ("hold", "id_hold", 500, ctxs), ("shiftR", "shift_R", 500, ["pri"])]
    if os.environ.get("P57_FRESH"):
        sets.append(("fresh", "fresh", None, ["pri"]))  # storage: fresh only in the primary context
    if lite:  # checkpoint series: calib and fresh in the primary context, common-Fisher Gram only
        sets = [t for t in sets if t[0] in ("calib", "fresh")]

    def prefix(net, x):
        h = e13.emb(net, x)
        for b in net.blocks[:k]:
            h = b(h)
        return h

    def suffix(net, o):
        for b in net.blocks[k + 1:]:
            o = b(o)
        return net.forward_head(net.norm(o)).float()

    res = dict(model=model, kind=kind, perm=p, run=run, slot=k, src=D.KEEP[run][k], contexts=ctxs, rho=RHO, data={}, cka={})
    lin_idx = [k, k - 1, 0]
    for sname, cname, n, cs in sets:
        x, y = S.cache(cname)
        x, y = (x[:n], y[:n]) if n else (x, y)
        for ctx in cs:
            inp, suf = (nat if ctx[0] == "p" or ctx == "nr" else m), (nat if ctx in ("pri", "rep") else m)
            gen = torch.Generator().manual_seed(1000 + k)
            acc = {a: [] for a in ("G_common", "G_euclid", "G_cand", "pred", "kl_base_ref")}
            raw = []
            for s in range(0, len(x), MB):
                xb = nrm(x[s:s + MB].to(S.dev).float().div_(255))
                h = prefix(inp, xb)
                zs = suffix(suf, h)
                Z = torch.stack([suffix(suf, nat.blocks[b](h)) for b in range(L)], 1)  # (n, 12, C)
                zb = suffix(suf, Bu(h))
                zs, Z, zb = zs.double(), Z.double(), zb.double()  # float64 Fisher products (PREREG37 EX1 lesson)
                E, eb = Z - zs[:, None], (zb - zs)[:, None]
                el = 0.5 * E[:, lin_idx[0]:lin_idx[0] + 1] + 0.3 * E[:, lin_idx[1]:lin_idx[1] + 1] + 0.2 * E[:, lin_idx[2]:lin_idx[2] + 1]
                own = Z[:, k] if suf is nat else zb
                pc = own.softmax(-1)[:, None]
                pe = torch.full_like(pc, 1.0 / pc.shape[-1])
                for nm, pp in (("G_common", pc), ("G_euclid", pe)):
                    A = torch.cat([E, eb, el, plant(E, k, pp, gen)], 1)
                    acc[nm].append(spec.fip(A, A, pp).cpu())
                gc = []
                for e_c, z_c in ((eb, zb), (E[:, k:k + 1], Z[:, k]), (el, zs + el[:, 0])):
                    A = torch.cat([E, e_c], 1)
                    gc.append(spec.fip(A, A, z_c.softmax(-1)[:, None]))
                acc["G_cand"].append(torch.stack(gc, 1).cpu())
                acc["pred"].append(torch.cat([zs.argmax(-1)[:, None], Z.argmax(-1), zb.argmax(-1)[:, None]], 1).cpu())
                lb = zb.log_softmax(-1)[:, None]
                acc["kl_base_ref"].append((lb.exp() * (lb - Z.log_softmax(-1))).sum(-1).cpu())
                if os.environ.get("P57_RAW") and sname == "calib" and ctx == "pri" and s < 100:
                    raw.append(torch.cat([E, eb, el], 1).cpu())
            d = {a: torch.cat(v).numpy() for a, v in acc.items()}
            if lite:
                d = {a: v for a, v in d.items() if a in ("G_common", "pred")}
            if (sname, ctx) != ("calib", "pri"):  # storage: float64 only where the 1e-6 known answers are checked
                d = {a: (v.astype(np.float32) if v.dtype == np.float64 else v) for a, v in d.items()}
            d["label"] = y.numpy()
            if raw:
                d["raw_effects"] = torch.cat(raw).numpy()
            res["data"][f"{sname}/{ctx}"] = d
            print(model, kind, p, k, sname, ctx, len(x), flush=True)
    if lite and os.environ.get("P57_FRESH"):  # network-level recovery state on fresh: top-1, KL to the teacher, agreement
        x, y = S.cache("fresh")
        kl, ag, ok = 0.0, 0, 0
        for s_ in range(0, len(x), 100):
            xb = nrm(x[s_:s_ + 100].to(S.dev).float().div_(255))
            lt, lm = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
            kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
            ag += int((lt.argmax(-1) == lm.argmax(-1)).sum())
            ok += int((lm.argmax(-1).cpu() == y[s_:s_ + 100]).sum())
        res["net_fresh"] = dict(kl=kl / len(x), agree=ag / len(x), top1=ok / len(x), n=len(x))
    xf, _ = S.cache("id_fit")
    for ctx, net in (("native_input", nat), ("recovered_input", m)):
        Fu, Fn = [], [[] for _ in range(L)]
        for s in range(0, 500, MB):
            h = prefix(net, nrm(xf[s:s + MB].to(S.dev).float().div_(255)))
            Fu.append(feat(Bu(h) - h))
            for b in range(L):
                Fn[b].append(feat(nat.blocks[b](h) - h))
        Fu = torch.cat(Fu)
        res["cka"][ctx] = [lin_cka(Fu, torch.cat(Fn[b])) for b in range(L)]
    os.makedirs(OUT, exist_ok=True)
    flat = {f"{key}/{a}": v for key, d in res.pop("data").items() for a, v in d.items()}
    np.savez(f"{OUT}/p57_{model}_{kind}_{p}_{k}{os.environ.get('P57_TAG', '')}.npz", meta=np.array(repr(res)), **flat)
    print("saved", {a: v.shape for a, v in list(flat.items())[:6]}, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
