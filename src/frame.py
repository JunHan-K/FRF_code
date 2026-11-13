"""Out-of-frame share nu (paper Eq. nu): the fraction of a block's pooled Fisher effect energy left after the best
combination of the reference effects with one coefficient vector shared by all images.
With pooled Gram pieces A = sum_n <e_b, e_c>, b = sum_n <e_b, e_u>, u = sum_n <e_u, e_u>, w solves A w = b by an
eigenvalue-truncated pseudo-inverse (eigenvalues below TAU * lambda_max dropped), i.e. the exact orthogonal projection
onto the span of the retained reference directions; nu = (u - 2 w'b + w'Aw) / u, clipped at 0.
TAU = 1e-8 is the preregistered cut-off. On the 297 recovered cells of the paper every reference Gram keeps all 12
directions (condition number <= 457), and TAU = 1e-10, no cut-off (lstsq) and the ridge estimator of spec.fit agree with
TAU = 1e-8 to 1e-15, 1e-15 and 3e-6 (src/check_nu.py reproduces this from the p57 outputs).
No torch import, so this module can be tested on CPU."""
import numpy as np

TAU = 1e-8


def fit(A, b, tau=TAU):
    """coefficients w and number of retained eigen-directions"""
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    lam, V = np.linalg.eigh(A)
    keep = lam > tau * lam.max()
    return V[:, keep] @ ((V[:, keep].T @ b) / lam[keep]), int(keep.sum())


def nu(A, b, u, tau=TAU):
    w, _ = fit(A, b, tau)
    return max(float((u - 2 * w @ b + w @ A @ w) / u), 0.0)


def nu_from_gram(G, n_ref=12, cand=-1, tau=TAU):
    """G: pooled Gram over [references..., candidate] (or with the candidate at index `cand`)"""
    G = np.asarray(G, dtype=np.float64)
    r = list(range(n_ref))
    return nu(G[np.ix_(r, r)], G[r, cand], G[cand, cand], tau)
