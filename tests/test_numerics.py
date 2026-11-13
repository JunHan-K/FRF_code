"""Out-of-frame share (src/frame.py) and simplex least squares (src/simplex.py) on known answers and degenerate input."""
import warnings

import numpy as np
import pytest

import frame as FR
from simplex import simplex_ls


def gram(E):
    """pooled Gram of effect vectors (rows of E)"""
    return E @ E.T


def test_nu_known_answers():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(12, 400))                         # 12 reference effects
    inside = 0.5 * R[3] + 0.3 * R[7] - 0.2 * R[0]
    out = rng.normal(size=400)
    out -= R.T @ np.linalg.lstsq(R.T, out, rcond=None)[0]   # orthogonal to every reference
    for e, want in ((inside, 0.0), (out, 1.0), (inside + out * np.linalg.norm(inside) / np.linalg.norm(out), 0.5)):
        G = gram(np.vstack([R, e]))
        assert FR.nu_from_gram(G) == pytest.approx(want, abs=1e-9)


def test_nu_rank_deficient_references_is_exact_projection():
    """duplicated / dependent references: the truncated pseudo-inverse still gives the exact residual (no ridge floor)"""
    rng = np.random.default_rng(1)
    R = rng.normal(size=(6, 300))
    R = np.vstack([R, R[:4] * 2.0, R[0] + R[1]])            # 11 references spanning 6 dimensions
    e = R[2] - 0.5 * R[5]
    A, b, u = gram(R), R @ e, e @ e
    assert FR.fit(A, b)[1] == 6
    assert FR.nu(A, b, u) == pytest.approx(0.0, abs=1e-9)


def test_simplex_vertex_and_interior():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(12, 200))
    A = gram(R)
    assert np.abs(simplex_ls(A, A[:, 3]) - np.eye(12)[3]).max() < 1e-6
    w = np.full(12, 1 / 12)
    assert np.abs(simplex_ls(A, A @ w) - w).max() < 1e-4


@pytest.mark.parametrize("A", [np.zeros((12, 12)), np.full((12, 12), np.nan)])
def test_simplex_degenerate_input(A):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        w = simplex_ls(A, np.zeros(12))
    assert np.allclose(w, 1 / 12)
