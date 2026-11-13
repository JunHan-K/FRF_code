"""PREREG76 (exploratory, post hoc): geometry-matched null for the per-image span. Is the large per-image reduction of the
shared-coefficient share a property of the Fisher geometry and of 12 free coefficients per image, or of the alignment between
real references and the candidate? One job = one ViT-S model; cells from out/p57 (L a,b; E a,b; N a), fresh[:500],
primary context (native input, native suffix, common Fisher metric p = softmax(z with B_k)).
Per image and cell, Gram matrices (float64) under p of:
  real  : [12 native reference effects, base, flip_1..flip_5]. flip_j (PRIMARY NULL): the candidate's class-centred effect
          with an independent random sign per (image, class), rescaled to the candidate's Fisher energy. The references,
          their Gram (spectrum, rank, cut-off), p and the metric are unchanged, and the candidate keeps its per-class
          magnitude profile; only the alignment between candidate and references is randomised.
  gauss : [12 iid Gaussian logit vectors scaled per image to the Fisher energy of the matching reference, base] (secondary)
  perm  : [the 12 reference effects of another image (previous image in the batch, cyclic), base] (secondary)
  foreign: [the 12 blocks of an independently trained model of the same width applied to the same h, base] (secondary)
Also per image: d_eff = (tr F)^2 / tr(F^2) with F = diag(p) - p p^T, i.e. (1 - sum p^2)^2 / (sum p^2 - 2 sum p^3 + (sum p^2)^2).
usage: p76.py <model>"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p9lib as P  # noqa: E402
import p11  # noqa: E402
import s12r  # noqa: E402
import spec  # noqa: E402
from p47 import e_paths  # noqa: E402
from p40 import FOREIGN  # noqa: E402

OUT = "out/p76"
MB, NIMG = 25, 500
CELLS = (("L", "a"), ("L", "b"), ("E", "a"), ("E", "b"), ("N", "a"))
SETS = ("real", "gauss", "perm", "foreign")
NFLIP = 5


def cells(model):
    out = []
    for kind, p in CELLS:
        for f in sorted(glob.glob(f"out/p57/p57_{model}_{kind}_{p}_*.npz")):
            out.append((kind, p, int(f.rsplit("_", 1)[1][:-4])))
    return out


@torch.no_grad()
def run_model(model):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    nat = P.nofuse(S.load(D.M).eval())
    fm = P.nofuse(S.load(s12r.MODELS[FOREIGN[model]]).eval())
    for q in list(nat.parameters()) + list(fm.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    x, _ = S.cache("fresh")
    x = x[:NIMG]
    loaded = {}
    for kind, p, k in cells(model):
        fn = f"{OUT}/p76_{model}_{kind}_{p}_{k}.npz"
        if os.path.exists(fn):
            continue
        if (kind, p) not in loaded:
            if kind == "L":
                run, pt, _ = p11.load_case(model, p)
            else:
                D.M = s12r.MODELS[model]
                run, pt, _ = e_paths(model, p, kind)
            m = P.nofuse(D.build(run).eval())
            m.load_state_dict(torch.load(pt, map_location=S.dev))
            for q in m.parameters():
                q.requires_grad_(False)
            loaded = {(kind, p): (m, run)}
        m, run = loaded[(kind, p)]
        Bu = m.blocks[k]
        gen = torch.Generator(device=S.dev).manual_seed(7600 + k)
        acc = {s: [] for s in SETS}
        effc = []

        def suffix(o):
            for b in nat.blocks[k + 1:]:
                o = b(o)
            return nat.forward_head(nat.norm(o)).float()

        for s in range(0, len(x), MB):
            h = e13.emb(nat, nrm(x[s:s + MB].to(S.dev).float().div_(255)))
            for b in nat.blocks[:k]:
                h = b(h)
            zs = suffix(h).double()
            Z = torch.stack([suffix(nat.blocks[b](h)) for b in range(12)], 1).double()
            Zf = torch.stack([suffix(fm.blocks[b](h)) for b in range(12)], 1).double()
            eb = (suffix(Bu(h)).double() - zs)[:, None]
            E, Ef = Z - zs[:, None], Zf - zs[:, None]
            pc = Z[:, k].softmax(-1)[:, None]
            g = torch.randn(E.shape, generator=gen, device=S.dev).double()
            en_e = torch.diagonal(spec.fip(E, E, pc), dim1=1, dim2=2)
            en_g = torch.diagonal(spec.fip(g, g, pc), dim1=1, dim2=2)
            g = g * (en_e / en_g.clamp_min(1e-300)).sqrt()[..., None]
            ebar = (pc * eb).sum(-1, keepdim=True)
            en_b = spec.fip(eb, eb, pc)[:, 0, 0]
            fl = []
            for _ in range(NFLIP):
                sg = torch.randint(0, 2, eb.shape, generator=gen, device=S.dev).double() * 2 - 1
                f = sg * (eb - ebar)
                en_f = spec.fip(f, f, pc)[:, 0, 0]
                fl.append(f * (en_b / en_f.clamp_min(1e-300)).sqrt()[:, None, None])
            for nm, A in (("real", torch.cat([E, eb] + fl, 1)), ("gauss", torch.cat([g, eb], 1)),
                          ("perm", torch.cat([E.roll(1, 0), eb], 1)), ("foreign", torch.cat([Ef, eb], 1))):
                acc[nm].append(spec.fip(A, A, pc).cpu())
            q2, q3 = (pc[:, 0] ** 2).sum(-1), (pc[:, 0] ** 3).sum(-1)
            effc.append(((1 - q2) ** 2 / (q2 - 2 * q3 + q2 ** 2)).cpu())
        d = {f"G_{nm}": torch.cat(v).numpy() for nm, v in acc.items()}
        d["deff"] = torch.cat(effc).numpy()
        np.savez(fn, model=model, kind=kind, perm=p, slot=k, src=D.KEEP[run][k], **d)
        print(model, kind, p, k, flush=True)


if __name__ == "__main__":
    run_model(sys.argv[1])
