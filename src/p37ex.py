"""PREREG37 EX1 recomputation in float64. The kdref run computed the ladder identities with
Fisher inner products of float32 inputs; per-image R1 - R2 then reaches 0.24 on images with a small denominator, while
in float64 the identity holds to ~1e-13. This recomputes R1 (direct), R2 (closed
form from the skip-origin Gram) and R3 (affine 2-reference least squares) per image in float64, same images as p14
(id_hold[:500]), for the controls that need no training: base, copy, scale_0.5/1.5/2.0.
usage: p37ex.py <model> <perm>"""
import glob
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
import spec  # noqa: E402

OUT = "out/p37ex"
SW = "."
NA, MB = 500, int(os.environ.get("P37EX_MB", 25))


def cen(z):
    return z - z.mean(-1, keepdim=True)


@torch.no_grad()
def main(model, p):
    os.makedirs(OUT, exist_ok=True)
    run, pt, _ = p11.load_case(model, p)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    nrm = S.Norm(nat).to(S.dev)
    L = len(nat.blocks)
    ks = sorted(int(f.split("_")[-2]) for f in glob.glob(f"{SW}/out/p25/{model}_{p}_*_kdref.json"))
    xh, _ = S.cache("id_hold")
    xh = torch.cat([nrm(xh[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])
    res = dict(model=model, perm=p, n=NA, cells={})
    for k in ks:
        i, j = D.KEEP[run][k], k
        Bj = nat.blocks[j]
        C = {"base": lambda h: m.blocks[k](h), "copy": lambda h: Bj(h)}
        for gm in (0.5, 1.5, 2.0):
            C[f"scale_{gm}"] = lambda h, gm=gm: h + gm * (Bj(h) - h)

        def suffix(o):
            for b in nat.blocks[j + 1:]:
                o = b(o)
            return nat.forward_head(nat.norm(o)).float()

        acc = {nm: dict(r1=[], r2=[], r3=[]) for nm in C}
        for s in range(0, NA, MB):
            h = e13.emb(m, xh[s:s + MB].to(S.dev))
            for b in m.blocks[:k]:
                h = b(h)
            z0 = cen(suffix(h)).double()
            E = torch.stack([cen(suffix(nat.blocks[b](h))).double() - z0 for b in range(L)], 1)
            for nm, f in C.items():
                zu = suffix(f(h))
                eu = (cen(zu).double() - z0)[:, None]
                pu = zu.double().softmax(-1)[:, None]
                G, c = spec.fip(E, E, pu), spec.fip(E, eu, pu)[..., 0]
                dI, dJ = E[:, i], E[:, j]
                r1 = (spec.fip((eu[:, 0] - dI)[:, None], (dJ - dI)[:, None], pu)[:, 0, 0]
                      / spec.fip((dJ - dI)[:, None], (dJ - dI)[:, None], pu)[:, 0, 0])
                Gii, Gjj, Gij = G[:, i, i], G[:, j, j], G[:, i, j]
                r2 = (c[:, j] - c[:, i] - Gij + Gii) / (Gii - 2 * Gij + Gjj)
                K = torch.zeros(len(G), 3, 3, dtype=torch.float64, device=G.device)
                K[:, 0, 0], K[:, 0, 1], K[:, 1, 0], K[:, 1, 1] = Gii, Gij, Gij, Gjj
                K[:, :2, 2], K[:, 2, :2] = 1.0, 1.0
                rhs = torch.stack([c[:, i], c[:, j], torch.ones_like(Gii)], 1)
                r3 = torch.linalg.solve(K, rhs)[:, 1]
                a = acc[nm]
                a["r1"].append(r1.cpu()), a["r2"].append(r2.cpu()), a["r3"].append(r3.cpu())
        cell = {}
        for nm, a in acc.items():
            r1, r2, r3 = (torch.cat(a[q]) for q in ("r1", "r2", "r3"))
            ok = torch.isfinite(r1) & torch.isfinite(r2) & torch.isfinite(r3)
            cell[nm] = dict(R1=float(r1[ok].mean()), max_abs_r1_r2=float((r1 - r2)[ok].abs().max()),
                            max_abs_r1_r3=float((r1 - r3)[ok].abs().max()), n_ok=int(ok.sum()))
        res["cells"][k] = cell
        print(model, p, k, {nm: (f"{v['max_abs_r1_r2']:.1e}", f"{v['max_abs_r1_r3']:.1e}") for nm, v in cell.items()}, flush=True)
    json.dump(res, open(f"{OUT}/p37ex_{model}_{p}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
