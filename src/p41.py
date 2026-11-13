"""PREREG41 (experiment A): equal importance, different relative behaviour. No training.
Recovered model m (perm run), slot k with i != j (same cells as PREREG39), B_u = m.blocks[k], B_i = native i, B_j = native j.
A1 (constructed): candidates c_b(h) = h + gamma_b (B_b(h) - h), b in {i, j}; gamma_b found by bisection on id_fit[:500] so that
   the removal KL of the candidate inside m (KL(p_cand || p_skip), m's own suffix) equals that of B_u within 2%.
   Reported on id_fit: removal KL, Fisher energy, R_nat (p14 m11 definition, native suffix). Truth on id_hold (all images):
   Dagree = agree(B_j-net) - agree(B_i-net), agreement = top-1 equal to the candidate net (as PREREG39).
A2 support: per-image id_hold agreement of the B_u net with the B_i and B_j nets (0/1 strings), for the image bootstrap of
   pair differences, plus Dagree recomputed (must equal p39.s).
usage: p41.py <model> <perm> [comma-separated slots]"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p9lib as P  # noqa: E402
import p11  # noqa: E402

OUT = "out/p41"
NA, MB = 500, int(os.environ.get("P41_MB", 25))
TOL, NBIS = 0.02, 30


def fcov(a, b, p):
    return ((p * a * b).sum(-1) - (p * a).sum(-1) * (p * b).sum(-1)).double()


def kl(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)


@torch.no_grad()
def main(model, p, slots=None):
    os.makedirs(OUT, exist_ok=True)
    run, pt, _ = p11.load_case(model, p)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    for q in list(nat.parameters()) + list(m.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(m.blocks)
    ks = [int(t) for t in slots.split(",")] if slots else [k for k in range(L) if D.KEEP[run][k] != k]
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])
    xh, _ = S.cache("id_hold")
    NH = len(xh)
    xh = torch.cat([nrm(xh[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NH, 100)])

    def prefix(x, k):
        h = e13.emb(m, x)
        for b in m.blocks[:k]:
            h = b(h)
        return h

    def suf_nat(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    def suf_meas(o, k):
        return P.tail_fn(m, o, k + 1)

    res = dict(model=model, perm=p, n_fit=NA, n_hold=NH, tol=TOL, cells={})
    for k in ks:
        i, j = D.KEEP[run][k], k
        Bi, Bj, Bu = nat.blocks[i], nat.blocks[j], m.blocks[k]
        # cache the slot input and the skip / B_u log-probs on id_fit (m's own suffix)
        H = [prefix(xf[s:s + MB].to(S.dev), k).cpu() for s in range(0, NA, MB)]
        LS = [suf_meas(h.to(S.dev), k).log_softmax(-1).cpu() for h in H]

        def rkl(f):  # removal KL of block function f inside m: mean KL(p_f || p_skip)
            return sum(float(kl(suf_meas(f(h.to(S.dev)), k).log_softmax(-1), ls.to(S.dev)).sum()) for h, ls in zip(H, LS)) / NA

        target = rkl(Bu)
        cand, info = {}, {}
        for nm, B in (("cand_i", Bi), ("cand_j", Bj)):
            f = lambda g: (lambda h, B=B, g=g: h + g * (B(h) - h))
            lo, hi = 0.0, 1.0
            while rkl(f(hi)) < target and hi < 64:
                hi *= 2
            g, val, it = hi, rkl(f(hi)), 0
            for it in range(NBIS):
                g = 0.5 * (lo + hi)
                val = rkl(f(g))
                if abs(val / max(target, 1e-12) - 1) <= TOL:
                    break
                lo, hi = (g, hi) if val < target else (lo, g)
            cand[nm] = f(g)
            info[nm] = dict(gamma=g, removal_kl=val, matched=bool(abs(val / max(target, 1e-12) - 1) <= TOL), n_bisect=it + 1)
        # fit-side descriptors: R_nat (m11), Fisher energy (native suffix, metric p_f, origin skip)
        fns = {"base": Bu, **cand}
        for nm, f in fns.items():
            Rs, en = [], 0.0
            for h in H:
                h = h.to(S.dev)
                zf, zi, zj, zs = suf_nat(f(h), k), suf_nat(Bi(h), k), suf_nat(Bj(h), k), suf_nat(h, k)
                pm = (0.5 * (zi + zj)).softmax(-1)
                Rs.append((fcov(zf - zi, zj - zi, pm) / fcov(zj - zi, zj - zi, pm).clamp_min(1e-12)).cpu())
                en += float(fcov(zf - zs, zf - zs, zf.softmax(-1)).sum())
            info.setdefault(nm, {}).update(R_nat=float(torch.cat(Rs).mean()), energy=en / NA)
        info["base"]["removal_kl"] = target
        # truth on id_hold: agreement of each candidate net with the B_i- and B_j-nets (m's suffix, gamma = 1 replacement)
        ag = {nm: {"i": [], "j": []} for nm in fns}
        for s in range(0, NH, MB):
            h = prefix(xh[s:s + MB].to(S.dev), k)
            ti, tj = suf_meas(Bi(h), k).argmax(-1), suf_meas(Bj(h), k).argmax(-1)
            for nm, f in fns.items():
                tf = suf_meas(f(h), k).argmax(-1)
                ag[nm]["i"].append((tf == ti).cpu())
                ag[nm]["j"].append((tf == tj).cpu())
        cell = dict(i=i, j=j)
        for nm in fns:
            ai, aj = torch.cat(ag[nm]["i"]).float(), torch.cat(ag[nm]["j"]).float()
            info[nm].update(agree_i=float(ai.mean()), agree_j=float(aj.mean()), Dagree=float(aj.mean() - ai.mean()))
            if nm == "base":
                info[nm]["per_image_hold"] = dict(agree_i="".join("1" if v else "0" for v in ai.tolist()),
                                                  agree_j="".join("1" if v else "0" for v in aj.tolist()))
        cell.update(info)
        res["cells"][k] = cell
        print(model, p, k, {nm: {a: (round(v, 4) if isinstance(v, float) else v) for a, v in info[nm].items() if a != "per_image_hold"}
                            for nm in fns}, flush=True)
    tag = f"_{slots.replace(',', '-')}" if slots else ""
    json.dump(res, open(f"{OUT}/p41_{model}_{p}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
