"""PREREG66 C: executable reference mixtures. For a cell (recovered network m, slot k, recovered block B_u) build
    B_w(h) = h + sum_b w_b (B_b(h) - h),  w on the simplex (w >= 0, sum w = 1), B_b = native blocks 0..11,
and insert it at slot k. Coefficients w from seven methods (all on calib = id_fit[:500]):
  single   : one-hot on the native block with the smallest KL(m || m with B_b) in the recovered network (D2)
  uniform  : 1/12
  hidden   : simplex least squares of the hidden update B_u(h) - h on the reference updates (native input, D1)
  euclid   : simplex least squares of the logit effect in the centred Euclidean metric (p57 calib/pri/G_euclid[:500])
  fisher   : the same in the common Fisher metric (p57 calib/pri/G_common[:500]) -- the FRF reconstruction
  exec_d1  : w = softmax(theta), 100 Adam steps (lr 0.1, minibatch 50) minimising KL(native net with B_u || with B_w), D1
  exec_d2  : the same inside the recovered network, KL(m || m with B_w), D2
Evaluation on fresh, in D2 (primary: the recovered network with B_w at slot k; KL to m, disagreement, top-1) and in D1
(native input and suffix; KL to the native network with B_u at k, disagreement). Error decomposition per context with a fixed
metric (D1: common Fisher, p = native network prediction; D2: p = m's prediction):
  e_B - e_mix = (e_B - sum_b w_b e_b) + (sum_b w_b e_b - e_mix), reporting the three energies and the cross term.
Known answers (P66_KA=1): one-hot mixtures reproduce single-block replacement logits; the simplex solver recovers a planted
vertex; with B_u := B_k (a copy) the hidden fit returns e_k and its D1 KL is ~0.
PREREG67 additions (before any p66 output): P66_SPARSE=1 adds supports of size 1..3 per metric (fisher, euclid, hidden)
chosen by the calibration objective; every run also evaluates all mixtures on ImageNet-R and ImageNet-A in D2.
usage: p66.py <model> <L|E|N> <perm> <slot>"""
import json
import os
import sys

import warnings

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p60 import parent_net  # noqa: E402

OUT = os.environ.get("P66_OUT", "out/p66")
P57DIRS = ("out/p57", "out/p65", "out/p66par")
dev = S.dev
NC, MB, STEPS, LR = 500, 50, 100, 0.1
KA = os.environ.get("P66_KA", "0").lower() not in ("", "0", "false", "no")


def simplex_ls(A, b):
    """min_w w'Aw - 2b'w on the simplex (A, b pooled Gram terms)"""
    s = np.trace(A) / len(A)
    A, b = A / s, b / s
    n = len(b)
    r = minimize(lambda w: w @ A @ w - 2 * b @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * A @ w - 2 * b, method="SLSQP",
                 bounds=[(0, 1)] * n, constraints=[dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(n))],
                 options=dict(maxiter=500, ftol=1e-14))
    w = np.clip(np.nan_to_num(r.x, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    valid = bool(np.isfinite(r.x).all() and w.sum() > 0)
    w = w / w.sum() if valid else np.full(n, 1.0 / n)
    if not (r.success and valid):  # convex QP: fall back to projected gradient on the simplex, keep the better solution
        warnings.warn(f"simplex_ls: SLSQP failed ({r.message}); using projected-gradient fallback")
        v, L = np.full(n, 1.0 / n), 2 * np.linalg.eigvalsh(A).max() + 1e-12
        for _ in range(20000):
            v = _proj_simplex(v - (2 * A @ v - 2 * b) / L)
        f = lambda u: u @ A @ u - 2 * b @ u
        if f(v) < f(w):
            w = v
    return w


def _proj_simplex(v):
    """Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(v) + 1) > css - 1)[0][-1]
    return np.maximum(v - (css[k] - 1) / (k + 1), 0)


def fisher_ip(a, b, p):  # a, b (n, C), p (n, C): Cov_p(a, b) per image
    return (p * a * b).sum(-1) - (p * a).sum(-1) * (p * b).sum(-1)


@torch.no_grad()
def main(model, kind, perm, slot):
    k = int(slot)
    D.M = s12r.MODELS[model]
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    run, m = parent_net(model, kind, perm)
    L = len(m.blocks)
    Bu = m.blocks[k]
    if KA:
        Bu = nat.blocks[k]
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
        return h + sum(float(w[b]) * (refs[b](h) - h) for b in range(L) if float(w[b]) != 0.0) if not torch.is_tensor(w) else \
            h + sum(w[b] * (refs[b](h) - h) for b in range(L))

    xc, _ = S.cache("id_fit")
    xc = xc[:NC]
    batches = [nrm(xc[s:s + MB].to(dev).float().div_(255)) for s in range(0, NC, MB)]
    # --- coefficient methods -------------------------------------------------------------------------------------------
    W = {"uniform": np.full(L, 1.0 / L)}
    kl_single = np.zeros(L)
    for xb in batches:
        h2 = prefix(m, xb)
        lt = suffix(m, Bu(h2)).log_softmax(-1)
        for b in range(L):
            lb = suffix(m, refs[b](h2)).log_softmax(-1)
            kl_single[b] += float((lt.exp() * (lt - lb)).sum(-1).sum())
    W["single"] = np.eye(L)[int(np.argmin(kl_single))]
    Gh = np.zeros((L + 1, L + 1))
    for xb in batches:
        h1 = prefix(nat, xb)
        V = torch.stack([(refs[b](h1) - h1).flatten(1).double() for b in range(L)] + [(Bu(h1) - h1).flatten(1).double()], 1)
        Gh += torch.einsum("nid,njd->ij", V, V).cpu().numpy()
    W["hidden"] = simplex_ls(Gh[:L, :L], Gh[:L, L])
    if not KA:
        z = None
        for d_ in P57DIRS:
            f = f"{d_}/p57_{model}_{kind}_{perm}_{k}.npz"
            if os.path.exists(f):
                z = np.load(f)
                break
        for nm, key in (("euclid", "calib/pri/G_euclid"), ("fisher", "calib/pri/G_common")):
            G = z[key][:NC].sum(0).astype(np.float64)
            W[nm] = simplex_ls(G[:L, :L], G[:L, L])
    if os.environ.get("P66_SPARSE") and not KA:  # PREREG67 Q2 sparse panel: best support of size 1..3 by the calib objective
        from itertools import combinations
        Gm = {"fisher": None, "euclid": None, "hidden": Gh}
        for nm, key in (("euclid", "calib/pri/G_euclid"), ("fisher", "calib/pri/G_common")):
            Gm[nm] = z[key][:NC].sum(0).astype(np.float64)
        for nm, G in Gm.items():
            A_, b_ = G[:L, :L], G[:L, L]
            for ks in (1, 2, 3):
                best = None
                for sup in combinations(range(L), ks):
                    sup = list(sup)
                    ws = simplex_ls(A_[np.ix_(sup, sup)], b_[sup]) if ks > 1 else np.ones(1)
                    obj = float(ws @ A_[np.ix_(sup, sup)] @ ws - 2 * b_[sup] @ ws)
                    if best is None or obj < best[0]:
                        best = (obj, sup, ws)
                w = np.zeros(L)
                w[best[1]] = best[2]
                W[f"{nm}_k{ks}"] = w
    for nm, ctx in (("exec_d1", nat), ("exec_d2", m)):
        theta = torch.zeros(L, device=dev, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=LR)
        H = [prefix(ctx, xb) for xb in batches]
        T = [suffix(ctx, Bu(h)).log_softmax(-1) for h in H]
        for t in range(STEPS):
            i = t % len(H)
            with torch.enable_grad():
                w = theta.softmax(0)
                lm = suffix(ctx, mix(H[i], w)).log_softmax(-1)
                loss = (T[i].exp() * (T[i] - lm)).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        W[nm] = theta.detach().softmax(0).cpu().double().numpy()
        res[f"{nm}_final_loss"] = float(loss)
        del H, T
    res["w"] = {nm: [round(float(v), 6) for v in w] for nm, w in W.items()}
    res["kl_single_calib"] = (kl_single / NC).tolist()
    # --- evaluation on fresh --------------------------------------------------------------------------------------------
    xf, yf = S.cache("fresh")
    ev = {nm: dict(d2_kl=0.0, d2_dis=0, d2_top1=0, d1_kl=0.0, d1_dis=0,
                   d1_E_tot=0.0, d1_E_rec=0.0, d1_E_exe=0.0, d1_E_x=0.0, d2_E_tot=0.0, d2_E_rec=0.0, d2_E_exe=0.0, d2_E_x=0.0)
          for nm in W}
    ka_onehot = []
    for s in range(0, len(xf), MB):
        xb = nrm(xf[s:s + MB].to(dev).float().div_(255))
        yb = yf[s:s + MB].to(dev)
        for ctx, tag in ((nat, "d1"), (m, "d2")):
            h = prefix(ctx, xb)
            zs = suffix(ctx, h).double()
            tB = suffix(ctx, Bu(h)).double()
            Z = torch.stack([suffix(ctx, refs[b](h)).double() for b in range(L)], 1)
            E = Z - zs[:, None]
            eB = tB - zs
            p = (Z[:, k] if tag == "d1" else tB).softmax(-1)  # D1 common Fisher (native net); D2 the recovered net
            lt = tB.log_softmax(-1)
            for nm, w in W.items():
                zm = suffix(ctx, mix(h, w)).double()
                if KA and tag == "d2" and nm == "single":
                    b = int(np.argmax(w))
                    ka_onehot.append(float((zm - Z[:, b]).abs().max()))
                lm = zm.log_softmax(-1)
                e = ev[nm]
                e[f"{tag}_kl"] += float((lt.exp() * (lt - lm)).sum(-1).sum())
                e[f"{tag}_dis"] += int((lm.argmax(-1) != lt.argmax(-1)).sum())
                if tag == "d2":
                    e["d2_top1"] += int((lm.argmax(-1) == yb).sum())
                wt = torch.tensor(w, device=dev, dtype=torch.float64)
                lin = (E * wt[None, :, None]).sum(1)
                emix = zm - zs
                r1, r2 = eB - lin, lin - emix
                e[f"{tag}_E_tot"] += float(fisher_ip(eB - emix, eB - emix, p).sum())
                e[f"{tag}_E_rec"] += float(fisher_ip(r1, r1, p).sum())
                e[f"{tag}_E_exe"] += float(fisher_ip(r2, r2, p).sum())
                e[f"{tag}_E_x"] += float(2 * fisher_ip(r1, r2, p).sum())
            if tag == "d1":
                e0 = float(fisher_ip(eB, eB, p).sum())
                res["d1_energy"] = res.get("d1_energy", 0.0) + e0
            else:
                res["d2_energy"] = res.get("d2_energy", 0.0) + float(fisher_ip(eB, eB, p).sum())
    if not KA:  # PREREG67 Q5: coefficients fixed on ID calibration, behaviour preservation under shift (D2, 1000-way output)
        for sname in ("shift_R", "shift_A"):
            xs_, _ = S.cache(sname)
            acc = {nm: [0.0, 0] for nm in W}
            for s in range(0, len(xs_), MB):
                xb = nrm(xs_[s:s + MB].to(dev).float().div_(255))
                h = prefix(m, xb)
                lt = suffix(m, Bu(h)).double().log_softmax(-1)
                for nm, w in W.items():
                    lm = suffix(m, mix(h, w)).double().log_softmax(-1)
                    acc[nm][0] += float((lt.exp() * (lt - lm)).sum(-1).sum())
                    acc[nm][1] += int((lm.argmax(-1) != lt.argmax(-1)).sum())
            res[sname] = {nm: dict(kl=v[0] / len(xs_), dis=v[1] / len(xs_)) for nm, v in acc.items()}
            res[f"{sname}_n"] = len(xs_)
    n = len(xf)
    for nm in ev:
        for q in ev[nm]:
            ev[nm][q] /= n
    res["d1_energy"] /= n
    res["d2_energy"] /= n
    res["eval"] = ev
    res["n_fresh"] = n
    if KA:
        A = np.random.default_rng(0).normal(size=(12, 12))
        A = A @ A.T
        res["ka_solver_vertex_err"] = float(np.abs(simplex_ls(A, A[:, 3]) - np.eye(12)[3]).max())
        res["ka_onehot_maxabs"] = max(ka_onehot) if ka_onehot else None
        res["ka_hidden_w_k"] = float(W["hidden"][k])
        res["ka_hidden_d1_kl"] = ev["hidden"]["d1_kl"]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/p66_{model}_{kind}_{perm}_{k}{'_ka' if KA else ''}.json", "w"), indent=1)
    print(json.dumps({q: v for q, v in res.items() if q != "eval"})[:1500], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
