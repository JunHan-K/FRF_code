"""Functional role spectrum estimator (PREREG34/35).
Model: e_u = s * sum_b w_b e_b + r,  w_b >= 0, sum_b w_b = 1, s >= 0, in the Fisher metric of a predictive p.
  composition w (what role), strength s (how strongly), novelty nu = |r|^2 / |e_u|^2 (function outside the reference set).
Solved as non-negative least squares v = s * w (v >= 0), then s = sum v, w = v / s.
Identifiability: reference blocks whose weights trade off against each other across image bootstraps are merged into
functional equivalence groups; the spectrum is reported over the groups (role resolution set by identifiability).
All inputs are per-image Fisher Gram pieces: Gx (N,L,L) = <e_b,e_c>, cx (N,L) = <e_b,e_u>, ux (N,) = <e_u,e_u>."""
import torch

RIDGE = 1e-4  # ridge * mean diagonal: a regularised estimate (slight floor for tiny effects); the out-of-frame share
# of the paper is frame.nu (truncated pseudo-inverse, exact projection; src/check_nu.py compares the two)
MERGE_CORR = -0.5  # merge two groups when their bootstrap weights correlate below this (trade-off = not identifiable)
NBOOT = 200


def fip(A, B, p):
    """Fisher inner products Cov_p(a, b) for A (N,L,C), B (N,M,C), p (N,1,C) -> (N,L,M) float64"""
    return (torch.einsum("nlc,nmc->nlm", A * p, B) - (A * p).sum(-1)[:, :, None] * (B * p).sum(-1)[:, None, :]).double()


def nnls(G, c, iters=500):
    """min_v 0.5 v'Gv - c'v, v >= 0 (batched, G positive definite): cyclic coordinate descent"""
    v = torch.zeros_like(c)
    d = G.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    for _ in range(iters):
        for b in range(c.shape[-1]):
            g = (G[..., b, :] * v).sum(-1) - c[..., b]
            v[..., b] = (v[..., b] - g / d[..., b]).clamp_min(0)
    return v


def ridge(G):
    return RIDGE * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)


def fit(G, c, u):
    """pooled or batched fit -> dict of tensors (v, s, w, nu, w_unconstrained, neg_mass)"""
    Gr = G + ridge(G)
    v = nnls(Gr, c)
    s = v.sum(-1)
    res = u - 2 * (v * c).sum(-1) + torch.einsum("...l,...lm,...m->...", v, G, v)
    wu = torch.linalg.solve(Gr, c[..., None])[..., 0]
    res_u = u - 2 * (wu * c).sum(-1) + torch.einsum("...l,...lm,...m->...", wu, G, wu)
    # nu (= nu_cone): residual of the non-negative role model; nu_span: residual of the unconstrained span fit.
    # nu_span = effect outside the reference span; nu - nu_span = cone violation (needs negative contributions)
    return dict(v=v, s=s, w=v / s.clamp_min(1e-12)[..., None], nu=res / u.clamp_min(1e-12),
                nu_span=res_u / u.clamp_min(1e-12),
                wu=wu, neg_mass=wu.clamp_max(0).abs().sum(-1) / wu.abs().sum(-1).clamp_min(1e-12))


def groups_from_boot(V):
    """V (B, L) bootstrap NNLS solutions -> list of index groups. Agglomerative: repeatedly merge the pair of groups
    whose summed weights have the most negative bootstrap correlation, while that correlation < MERGE_CORR."""
    grp = [[b] for b in range(V.shape[1])]
    while len(grp) > 1:
        W = torch.stack([V[:, g].sum(1) for g in grp], 1)
        W = W - W.mean(0)
        sd = W.pow(2).mean(0).sqrt()
        live = sd > 1e-9
        C = (W.T @ W) / len(W) / (sd[:, None] * sd[None, :]).clamp_min(1e-30)
        C = C.masked_fill(~(live[:, None] & live[None, :]), 1.0).fill_diagonal_(1.0)
        a = int(C.argmin()) // C.shape[1]
        b = int(C.argmin()) % C.shape[1]
        if float(C[a, b]) >= MERGE_CORR:
            break
        grp[a] = sorted(grp[a] + grp[b])
        del grp[b]
    return grp


def analyze(Gx, cx, ux, names, seed=0):
    """full report for one measured block. names: label of each reference (e.g. native block index)."""
    Gs, cs, us = Gx.sum(0), cx.sum(0), ux.sum()
    P = fit(Gs, cs, us)
    I = fit(Gx, cx, ux)  # per image
    gen = torch.Generator().manual_seed(seed)
    n = Gx.shape[0]
    cnt = torch.zeros(NBOOT, n, dtype=Gx.dtype)
    idx = torch.randint(0, n, (NBOOT, n), generator=gen)
    cnt.scatter_add_(1, idx, torch.ones_like(idx, dtype=Gx.dtype))
    Vb = fit(torch.einsum("bn,nlm->blm", cnt, Gx), cnt @ cx, cnt @ ux)["v"]
    grp = groups_from_boot(Vb)
    dg = Gs.diagonal().clamp_min(1e-30).rsqrt()
    Gn = dg[:, None] * Gs * dg[None, :]  # Fisher cosine matrix of the references
    Vc = Vb - Vb.mean(0)
    sd = Vc.pow(2).mean(0).sqrt()
    bcorr = (Vc.T @ Vc) / len(Vc) / (sd[:, None] * sd[None, :]).clamp_min(1e-30)
    gw = [float(P["w"][g].sum()) for g in grp]
    gb = torch.stack([Vb[:, g].sum(1) / Vb.sum(1).clamp_min(1e-12) for g in grp], 1)
    return dict(
        w=[round(float(x), 4) for x in P["w"]], s=float(P["s"]), nu=float(P["nu"]), nu_span=float(P["nu_span"]),
        gram=[[float(x) for x in row] for row in Gs], fisher_cos=[[round(float(x), 4) for x in row] for row in Gn],
        sv_min=float(torch.linalg.svdvals(Gn).min()), boot_corr=[[round(float(x), 3) for x in row] for row in bcorr],
        v_unconstrained=[round(float(x), 4) for x in P["wu"]], neg_mass=float(P["neg_mass"]),
        cond=float(torch.linalg.cond(Gn)),
        groups=[[names[b] for b in g] for g in grp], group_w=[round(x, 4) for x in gw],
        group_w_ci=[[round(float(x), 4) for x in torch.quantile(gb[:, t], torch.tensor([0.025, 0.975], dtype=gb.dtype))]
                    for t in range(len(grp))],
        pi=dict(s=I["s"].float(), nu=I["nu"].float(), nu_span=I["nu_span"].float(), w=I["w"].float()))


def synthetic(Gx, j, gammas=(0.5, 1.0, 1.5, 2.0), betas=(0.0, 0.25, 0.5, 0.75)):
    """known-answer tests of the estimator on the cell's own reference geometry (exact ground truth):
    strength: e_u = gamma * e_j            -> w_j (or j's group) = 1, s = gamma, nu = 0
    novelty : e_u = (1-beta) e_j + beta q, q Fisher-orthogonal to every reference and |q| = |e_j| per image
              -> w_j = 1, s = 1 - beta, nu = beta^2 / ((1-beta)^2 + beta^2)"""
    out = dict(strength={}, novelty={})
    Gjj = Gx[:, j, j]
    for g in gammas:
        P = fit(Gx.sum(0), g * Gx[:, :, j].sum(0), (g * g * Gjj).sum())
        out["strength"][g] = dict(wj=float(P["w"][j]), s=float(P["s"]), nu=float(P["nu"]))
    for bt in betas:
        P = fit(Gx.sum(0), (1 - bt) * Gx[:, :, j].sum(0), (((1 - bt) ** 2 + bt ** 2) * Gjj).sum())
        out["novelty"][bt] = dict(wj=float(P["w"][j]), s=float(P["s"]), nu=float(P["nu"]),
                                  nu_true=bt ** 2 / ((1 - bt) ** 2 + bt ** 2))
    return out
