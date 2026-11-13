"""PREREG72: does the context-transfer gap measured on split A predict, beyond initial mismatch, which reference
mixtures benefit from re-estimating their coefficients in the recovered context? One job = one PREREG66 cell (recovered network
m, slot k, recovered block B_u); mixture block B_w(h) = h + sum_b w_b (B_b(h) - h), w on the simplex, B_b = native blocks.
Splits:   A = id_fit[0:500] (diagnosis), B = id_fit[500:1000] (coefficient fitting), C = fresh (final evaluation, D2 only).
Contexts: D1 = (native input, native suffix) = p57 'pri';  D2 = (recovered input, recovered suffix) = p57 'rr'.
Metrics:  fisher  (D1: p = native prediction with B_k; D2: p = m's prediction with B_u)  and  centred (p uniform, = p57 G_euclid).
Effects e = z(X) - z(skip) in the context; Gram of [12 references, e_B] pooled over images (float64).
Diagnosis on A (per metric), with W_A = simplex least squares of the D1 Gram on A:
  r_dict = simplex residual energy / |e_B|^2 (D1);  r_exe1 = |e_B - e_mix|^2 / |e_B|^2 executing B_{W_A} in D1;
  r_d2 = the same in D2;  G_dict = log r_dict, G_suffix = log r_exe1 - log r_dict, G_context = log r_d2 - log r_exe1;
  K_NA = log mean KL(m || m with B_{W_A}) on A.
Policies fitted on B only (identical solver, init, tolerance): WN = simplex LS of the D1 Gram on B, WR = of the D2 Gram on B.
W0 = the PREREG66 'fisher' weights (fitted on id_fit[:500], which overlaps A; auxiliary only).
Evaluation on C in D2 for every policy and metric: mean KL(m || m with B_W), disagreement with m.
P72_KA=1 (known answers, no outcome): (a) B_u := native B_k -> WN and WR put >= 0.99 on k and their D2 KL on C[:200] ~ 0;
(b) with the real B_u, the recomputed A-split Grams reproduce p57 calib/pri/G_common, calib/rr/G_common and the G_euclid
counterparts on id_fit[:500]; (c) the simplex solver recovers a planted vertex.
usage: p72.py <model> <L|E|N> <perm> <slot>"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p60 import parent_net  # noqa: E402
from p66 import simplex_ls, P57DIRS  # noqa: E402

OUT = os.environ.get("P72_OUT", "out/p72")
P66OUT = "out/p66"
dev = S.dev
MB = 50
KA = os.environ.get("P72_KA", "0").lower() not in ("", "0", "false", "no")
METRICS = ("fisher", "centred")


def gram(A, p):  # A (n, m, C) float64, p (n, C): pooled sum_n Cov_p(A_i, A_j)
    pa = A * p[:, None]
    mu = pa.sum(-1)
    return (torch.einsum("nic,njc->ij", pa, A) - torch.einsum("ni,nj->ij", mu, mu)).cpu().numpy()


def energy(r, p):  # pooled sum_n Cov_p(r, r), r (n, C)
    pr = r * p
    return float(((pr * r).sum(-1) - pr.sum(-1) ** 2).sum())


def resid(G, w):
    L = len(w)
    return float(G[L, L] - 2 * G[:L, L] @ w + w @ G[:L, :L] @ w)


@torch.no_grad()
def main(model, kind, perm, slot):
    k = int(slot)
    D.M = s12r.MODELS[model]
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    run, m = parent_net(model, kind, perm)
    L = len(m.blocks)
    refs = [nat.blocks[b] for b in range(L)]
    res = dict(model=model, kind=kind, perm=perm, slot=k, src=D.KEEP[run][k], ka=KA)

    def prefix(net, x):
        h = e13.emb(net, x)
        for b in net.blocks[:k]:
            h = b(h)
        return h

    def suffix(net, o):
        for b in net.blocks[k + 1:]:
            o = b(o)
        return net.forward_head(net.norm(o)).float()

    def mix(h, w):
        return h + sum(float(w[b]) * (refs[b](h) - h) for b in range(L) if float(w[b]) != 0.0)

    def pm(metric, ctx_tag, Z, tB):  # metric distribution per image
        if metric == "centred":
            return torch.full_like(tB, 1.0 / tB.shape[-1])
        return (Z[:, k] if ctx_tag == "d1" else tB).softmax(-1)

    xf, _ = S.cache("id_fit")
    split = {"A": xf[0:500], "B": xf[500:1000]}
    ctxs = (("d1", nat), ("d2", m))

    def grams(Bu, sets):
        G = {(s, c, mt): np.zeros((L + 1, L + 1)) for s in sets for c, _ in ctxs for mt in METRICS}
        for s in sets:
            x = split[s]
            for i in range(0, len(x), MB):
                xb = nrm(x[i:i + MB].to(dev).float().div_(255))
                for tag, ctx in ctxs:
                    h = prefix(ctx, xb)
                    zs = suffix(ctx, h).double()
                    Z = torch.stack([suffix(ctx, refs[b](h)).double() for b in range(L)], 1)
                    tB = suffix(ctx, Bu(h)).double()
                    A = torch.cat([Z - zs[:, None], (tB - zs)[:, None]], 1)
                    for mt in METRICS:
                        G[(s, tag, mt)] += gram(A, pm(mt, tag, Z, tB))
        return G

    def eval_c(Bu, W, n=None):  # D2 on C: mean KL(m || m with B_W), disagreement
        xc, _ = S.cache("fresh")
        xc = xc[:n] if n else xc
        acc = {nm: [0.0, 0] for nm in W}
        for i in range(0, len(xc), MB):
            xb = nrm(xc[i:i + MB].to(dev).float().div_(255))
            h = prefix(m, xb)
            lt = suffix(m, Bu(h)).double().log_softmax(-1)
            for nm, w in W.items():
                lm = suffix(m, mix(h, w)).double().log_softmax(-1)
                acc[nm][0] += float((lt.exp() * (lt - lm)).sum(-1).sum())
                acc[nm][1] += int((lm.argmax(-1) != lt.argmax(-1)).sum())
        return {nm: dict(kl=v[0] / len(xc), dis=v[1] / len(xc)) for nm, v in acc.items()}, len(xc)

    if KA:
        ka = {}
        # (a) planted copy: B_u := native B_k
        Gk = grams(nat.blocks[k], ("B",))
        Wk = {}
        for mt in METRICS:
            for pol, c in (("WN", "d1"), ("WR", "d2")):
                G = Gk[("B", c, mt)]
                Wk[f"{pol}_{mt}"] = simplex_ls(G[:L, :L], G[:L, L])
                ka[f"ka_w_k_{pol}_{mt}"] = float(Wk[f"{pol}_{mt}"][k])
        ev, _ = eval_c(nat.blocks[k], Wk, n=200)
        ka["ka_copy_d2_kl_max"] = max(v["kl"] for v in ev.values())
        # (b) Gram reproduction against p57 on id_fit[:500] with the real B_u
        Ga = grams(m.blocks[k], ("A",))
        z = next(np.load(f"{d_}/p57_{model}_{kind}_{perm}_{k}.npz") for d_ in P57DIRS
                 if os.path.exists(f"{d_}/p57_{model}_{kind}_{perm}_{k}.npz"))
        for c, ctxname in (("d1", "pri"), ("d2", "rr")):
            for mt, key in (("fisher", "G_common"), ("centred", "G_euclid")):
                if f"calib/{ctxname}/{key}" not in z.files:  # new parents store p57 'pri' G_common only (lite)
                    ka[f"ka_gram_{c}_{mt}_relmax"] = None
                    continue
                ref = z[f"calib/{ctxname}/{key}"][:500, :L + 1, :L + 1].astype(np.float64).sum(0)
                ka[f"ka_gram_{c}_{mt}_relmax"] = float(np.abs(Ga[("A", c, mt)] - ref).max() / np.abs(ref).max())
        # (c) solver
        R = np.random.default_rng(0).normal(size=(12, 12))
        R = R @ R.T
        ka["ka_solver_vertex_err"] = float(np.abs(simplex_ls(R, R[:, 3]) - np.eye(12)[3]).max())
        ka["pass_"] = bool(min(v for q, v in ka.items() if q.startswith("ka_w_k")) >= 0.99
                           and ka["ka_copy_d2_kl_max"] <= 1e-4
                           and ka["ka_gram_d1_fisher_relmax"] is not None
                           and all(ka[f"ka_gram_{c}_{mt}_relmax"] is None or ka[f"ka_gram_{c}_{mt}_relmax"] <= tol
                                   for c, tol in (("d1", 1e-4), ("d2", 1e-3)) for mt in ("fisher", "centred"))
                           and ka["ka_solver_vertex_err"] <= 1e-6)
        res.update(ka)
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(f"{OUT}/p72_{model}_{kind}_{perm}_{k}_ka.json", "w"), indent=1)
        print(json.dumps(res), flush=True)
        return

    Bu = m.blocks[k]
    G = grams(Bu, ("A", "B"))
    res["gram"] = {f"{s}/{c}/{mt}": g.tolist() for (s, c, mt), g in G.items()}
    # --- diagnosis on A ------------------------------------------------------------------------------------------------
    WA = {}
    diag = {}
    for mt in METRICS:
        Gd1 = G[("A", "d1", mt)]
        WA[mt] = simplex_ls(Gd1[:L, :L], Gd1[:L, L])
        diag[mt] = dict(eB_d1=float(Gd1[L, L]), eB_d2=float(G[("A", "d2", mt)][L, L]), res_dict=resid(Gd1, WA[mt]),
                        exe_d1=0.0, exe_d2=0.0, kl_d2=0.0)
    xa = split["A"]
    for i in range(0, len(xa), MB):
        xb = nrm(xa[i:i + MB].to(dev).float().div_(255))
        for tag, ctx in ctxs:
            h = prefix(ctx, xb)
            zs = suffix(ctx, h).double()
            tB = suffix(ctx, Bu(h)).double()
            zk = suffix(ctx, refs[k](h)).double() if tag == "d1" else None
            lt = tB.log_softmax(-1)
            for mt in METRICS:
                zm = suffix(ctx, mix(h, WA[mt])).double()
                p = torch.full_like(tB, 1.0 / tB.shape[-1]) if mt == "centred" else (zk if tag == "d1" else tB).softmax(-1)
                diag[mt][f"exe_{tag}"] += energy((tB - zs) - (zm - zs), p)
                if tag == "d2":
                    lm = zm.log_softmax(-1)
                    diag[mt]["kl_d2"] += float((lt.exp() * (lt - lm)).sum(-1).sum())
    for mt in METRICS:
        d = diag[mt]
        d["kl_d2"] /= len(xa)
        d["r_dict"] = d["res_dict"] / d["eB_d1"]
        d["r_exe1"] = d["exe_d1"] / d["eB_d1"]
        d["r_d2"] = d["exe_d2"] / d["eB_d2"]
        d["G_dict"] = float(np.log(max(d["r_dict"], 1e-12)))
        d["G_suffix"] = float(np.log(max(d["r_exe1"], 1e-12)) - np.log(max(d["r_dict"], 1e-12)))
        d["G_context"] = float(np.log(max(d["r_d2"], 1e-12)) - np.log(max(d["r_exe1"], 1e-12)))
        d["K_NA"] = float(np.log(max(d["kl_d2"], 1e-12)))
    res["diag_A"] = diag
    res["W_A"] = {mt: w.tolist() for mt, w in WA.items()}
    # --- policies on B -------------------------------------------------------------------------------------------------
    W = {}
    for mt in METRICS:
        for pol, c in (("WN", "d1"), ("WR", "d2")):
            Gb = G[("B", c, mt)]
            W[f"{pol}_{mt}"] = simplex_ls(Gb[:L, :L], Gb[:L, L])
    f66 = f"{P66OUT}/p66_{model}_{kind}_{perm}_{k}.json"
    if os.path.exists(f66):
        W["W0_fisher"] = np.array(json.load(open(f66))["w"]["fisher"], dtype=np.float64)
    res["w"] = {nm: [round(float(v), 8) for v in w] for nm, w in W.items()}
    # --- evaluation on C (D2) ------------------------------------------------------------------------------------------
    res["eval_C"], res["n_C"] = eval_c(Bu, W)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/p72_{model}_{kind}_{perm}_{k}.json", "w"), indent=1)
    print("done", model, kind, perm, k, flush=True)  # no outcome printed


if __name__ == "__main__":
    main(*sys.argv[1:])
