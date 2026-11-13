"""Numerical check of the out-of-frame share on the stored p57 Grams: for every cell, nu with the preregistered cut-off
(frame.TAU = 1e-8) against TAU = 1e-10, no cut-off (lstsq) and the ridge estimator of spec.fit (1e-4 x mean diagonal);
also the number of retained directions and the condition number of the reference Gram. Reads fresh/pri/G_common
(calib/pri/G_common if the run had no fresh set).
usage: python src/check_nu.py [out/p57]"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frame as FR  # noqa: E402


def ridge_nu(A, b, u, r=1e-4):
    w = np.linalg.solve(A + r * np.mean(np.diag(A)) * np.eye(len(A)), b)
    return float((u - 2 * w @ b + w @ A @ w) / u)


def lstsq_nu(A, b, u):
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    return float((u - 2 * w @ b + w @ A @ w) / u)


def main(d="out/p57"):
    rows = []
    for f in sorted(glob.glob(f"{d}/p57_*.npz")):
        z = np.load(f)
        key = "fresh/pri/G_common" if "fresh/pri/G_common" in z.files else "calib/pri/G_common"
        G = z[key].astype(np.float64).sum(0)
        r = list(range(12))
        A, b, u = G[np.ix_(r, r)], G[r, 12], G[12, 12]
        lam = np.linalg.eigvalsh(A)
        rows.append((FR.nu(A, b, u), FR.nu(A, b, u, 1e-10), lstsq_nu(A, b, u), ridge_nu(A, b, u),
                     FR.fit(A, b)[1], lam.max() / lam.min()))
    if not rows:
        sys.exit(f"no p57 outputs in {d}")
    a = np.array(rows)
    print(f"cells {len(a)}; retained directions min {int(a[:, 4].min())} of 12; max condition number {a[:, 5].max():.3g}")
    for j, nm in ((1, "tau 1e-10"), (2, "lstsq"), (3, "ridge 1e-4")):
        dd = np.abs(a[:, j] - a[:, 0])
        print(f"{nm:>10} vs tau 1e-8: max |diff| {dd.max():.2e}, median {np.median(dd):.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:])
