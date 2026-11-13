"""PREREG45 (experiment A', replaces the PREREG41 construction): natural candidates at a fixed slot. No training, no scaling.
Recovered model m (perm run), slot k with i != j (PREREG39 cells), B_i = native i, B_j = native j, h = m's state entering k.
Candidates: every native block B_b (b = 0..L-1) inserted at slot k of m (gamma = 1), plus base (B_u = m.blocks[k]).
  b in {i, j} are the known-answer copies (KA1); the panel for the pairs is b not in {i, j}.
Per candidate on id_fit[:500]:
  removal KL (mean KL(p_cand || p_skip), m's own suffix, as p41), Fisher energy (native suffix, metric p_cand, origin skip),
  R_nat (p14 m11: native suffix, midpoint Fisher p_m, mean of per-image ratios),
  direct baselines as p39: agree_fit_diff = agree(B_j-net) - agree(B_i-net), kl_diff = KL(p_c || p_i) - KL(p_c || p_j) (m's suffix).
Truth on id_hold (all images): per-image top-1 agreement of the candidate net with the B_i- and B_j-nets (0/1 strings), Dagree.
usage: p45.py <model> <perm> [comma-separated slots]"""
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

OUT = "out/p45"
NA, MB = 500, int(os.environ.get("P45_MB", 25))


def fcov(a, b, p):
    return ((p * a * b).sum(-1) - (p * a).sum(-1) * (p * b).sum(-1)).double()


def kl(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)


def bits(t):
    return "".join("1" if v else "0" for v in t.tolist())


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

    res = dict(model=model, perm=p, n_fit=NA, n_hold=NH, cells={})
    for k in ks:
        i, j = D.KEEP[run][k], k
        Bi, Bj = nat.blocks[i], nat.blocks[j]
        fns = {"base": m.blocks[k], **{f"nat{b}": nat.blocks[b] for b in range(L)}}
        acc = {nm: dict(rkl=0.0, en=0.0, R=[], ai=0, aj=0, kli=0.0, klj=0.0) for nm in fns}
        for s in range(0, NA, MB):
            h = prefix(xf[s:s + MB].to(S.dev), k)
            ls = suf_meas(h, k).log_softmax(-1)
            li, lj = suf_meas(Bi(h), k).log_softmax(-1), suf_meas(Bj(h), k).log_softmax(-1)
            zi, zj, zs = suf_nat(Bi(h), k), suf_nat(Bj(h), k), suf_nat(h, k)
            pm = (0.5 * (zi + zj)).softmax(-1)
            den = fcov(zj - zi, zj - zi, pm).clamp_min(1e-12)
            for nm, f in fns.items():
                o = f(h)
                lc = suf_meas(o, k).log_softmax(-1)
                zf = suf_nat(o, k)
                a = acc[nm]
                a["rkl"] += float(kl(lc, ls).sum())
                a["en"] += float(fcov(zf - zs, zf - zs, zf.softmax(-1)).sum())
                a["R"].append((fcov(zf - zi, zj - zi, pm) / den).cpu())
                a["ai"] += int((lc.argmax(-1) == li.argmax(-1)).sum())
                a["aj"] += int((lc.argmax(-1) == lj.argmax(-1)).sum())
                a["kli"] += float(kl(lc, li).sum())
                a["klj"] += float(kl(lc, lj).sum())
        info = {nm: dict(removal_kl=a["rkl"] / NA, energy=a["en"] / NA, R_nat=float(torch.cat(a["R"]).mean()),
                         agree_fit_diff=(a["aj"] - a["ai"]) / NA, kl_diff=(a["kli"] - a["klj"]) / NA) for nm, a in acc.items()}
        ag = {nm: {"i": [], "j": []} for nm in fns}
        for s in range(0, NH, MB):
            h = prefix(xh[s:s + MB].to(S.dev), k)
            ti, tj = suf_meas(Bi(h), k).argmax(-1), suf_meas(Bj(h), k).argmax(-1)
            for nm, f in fns.items():
                tf = suf_meas(f(h), k).argmax(-1)
                ag[nm]["i"].append((tf == ti).cpu())
                ag[nm]["j"].append((tf == tj).cpu())
        for nm in fns:
            ai, aj = torch.cat(ag[nm]["i"]), torch.cat(ag[nm]["j"])
            info[nm].update(agree_i=float(ai.float().mean()), agree_j=float(aj.float().mean()),
                            Dagree=float(aj.float().mean() - ai.float().mean()),
                            per_image_hold=dict(agree_i=bits(ai), agree_j=bits(aj)))
        res["cells"][k] = dict(i=i, j=j, cands=info)
        print(model, p, k, {nm: (round(v["removal_kl"], 4), round(v["R_nat"], 3), round(v["Dagree"], 4)) for nm, v in info.items()},
              flush=True)
    tag = f"_{slots.replace(',', '-')}" if slots else ""
    json.dump(res, open(f"{OUT}/p45_{model}_{p}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
