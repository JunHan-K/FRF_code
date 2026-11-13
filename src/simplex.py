"""Least squares on the probability simplex (p66 / p68 / p72 block mixtures). numpy / scipy only."""
import warnings

import numpy as np
from scipy.optimize import minimize


def simplex_ls(A, b):
    """min_w w'Aw - 2b'w on the simplex (A, b pooled Gram terms). A must be finite and symmetric positive semidefinite;
    a zero or non-finite scale (degenerate Gram) gives uniform weights with a warning."""
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(b)
    s = np.trace(A) / n
    if not (np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(s) and s > 0):
        warnings.warn("simplex_ls: degenerate input (zero or non-finite Gram); returning uniform weights")
        return np.full(n, 1.0 / n)
    A, b = A / s, b / s
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
