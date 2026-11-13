"""PREREG56: robustness of the out-of-frame share to measurement choices, Fisher energy vs KL, and a CKA baseline.
Cells and inputs as p47 (L / E networks, slots 6..11, the network's own state entering k, native suffix after k).
Per cell, for the recovered block B_u (base):
  nu_fisher  : as p47 (metric p_candidate, id_fit[:500]); known answer = p47 base nu_span12 (within 2e-3)
  nu_euclid  : same fit, centred Euclidean metric on logits (p = uniform, i.e. (I - 11'/C) / C)
  nu_n125, nu_n250 : Fisher fit on the first 125 / 250 of the 500 images
  nu_shiftR  : Fisher fit on shift_R[:500] (ImageNet-R) inputs instead of id_fit
  energy / KL: mean 0.5 e'F(p_u)e, mean KL(p_u || p_skip), mean KL(p_skip || p_u)
  known answers: native copy B_k -> nu 0 in the Euclidean and the ImageNet-R fits (ka_copy_*)
  CKA        : linear CKA between block updates (B(h) - h), features = [CLS update, mean patch-token update], id_fit[:500];
               cka_max = max over the 12 native blocks, cka_k (slot native), cka_i (source block)
usage: p56.py <model> <L|E> <perm>"""
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
import s12r  # noqa: E402
import spec  # noqa: E402
from p47 import e_paths  # noqa: E402

OUT = "out/p56"
NA, MB = 500, 25


def lin_cka(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    return float((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm()).clamp_min(1e-30))


def feat(d):
    return torch.cat([d[:, 0], d[:, 1:].mean(1)], 1).double().cpu()


@torch.no_grad()
def main(model, kind, p):
    os.makedirs(OUT, exist_ok=True)
    if kind == "L":
        run, pt, _ = p11.load_case(model, p)
    else:
        D.M = s12r.MODELS[model]
        run, pt, _ = e_paths(model, p, kind)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    for q in list(nat.parameters()) + list(m.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(m.blocks)
    ks = [k for k in range(6, L) if (D.KEEP[run][k] != k if kind == "L" else D.KEEP[run][k] == k)]
    xs = {}
    for name in ("id_fit", "shift_R"):
        x, _ = S.cache(name)
        xs[name] = torch.cat([nrm(x[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])

    def prefix(x, k):
        h = e13.emb(m, x)
        for b in m.blocks[:k]:
            h = b(h)
        return h

    def suf_nat(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    def fitted(a):
        return float(spec.fit(a["G"], a["c"], torch.tensor(a["u"], dtype=torch.float64))["nu_span"])

    res = dict(model=model, kind=kind, perm=p, run=run, cells={})
    for k in ks:
        i, Bu = D.KEEP[run][k], m.blocks[k]
        new = lambda: dict(G=torch.zeros(L, L, dtype=torch.float64), c=torch.zeros(L, dtype=torch.float64), u=0.0)  # noqa: E731
        acc = {nm: new() for nm in ("fisher", "euclid", "shiftR", "euclid_copy", "shiftR_copy")}
        snap, en, klf, klr = {}, 0.0, 0.0, 0.0
        Fu, Fn = [], [[] for _ in range(L)]
        for src in ("id_fit", "shift_R"):
            for s in range(0, NA, MB):
                h = prefix(xs[src][s:s + MB].to(S.dev), k)
                zs = suf_nat(h, k)
                E = torch.stack([suf_nat(nat.blocks[b](h), k) - zs for b in range(L)], 1)
                ou = Bu(h)
                z = suf_nat(ou, k)
                pz = z.softmax(-1)[:, None]
                e = (z - zs)[:, None]
                ms = [("fisher", pz), ("euclid", torch.full_like(pz, 1.0 / pz.shape[-1]))] if src == "id_fit" else [("shiftR", pz)]
                for nm, pp in ms:
                    a = acc[nm]
                    a["G"] += spec.fip(E, E, pp).sum(0).cpu()
                    a["c"] += spec.fip(E, e, pp)[..., 0].sum(0).cpu()
                    a["u"] += float(spec.fip(e, e, pp)[:, 0, 0].sum())
                    ec, a = E[:, k:k + 1], acc[nm + "_copy"] if nm != "fisher" else None  # known answer: native copy -> nu 0
                    if a is not None:
                        a["G"] += spec.fip(E, E, pp).sum(0).cpu()
                        a["c"] += spec.fip(E, ec, pp)[..., 0].sum(0).cpu()
                        a["u"] += float(spec.fip(ec, ec, pp)[:, 0, 0].sum())
                if src != "id_fit":
                    continue
                n_done = s + MB
                if n_done in (125, 250):
                    snap[n_done] = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in acc["fisher"].items()}
                lu, l0 = z.log_softmax(-1), zs.log_softmax(-1)
                en += float(0.5 * spec.fip(e, e, pz)[:, 0, 0].sum())
                klf += float((lu.exp() * (lu - l0)).sum(-1).sum())
                klr += float((l0.exp() * (l0 - lu)).sum(-1).sum())
                Fu.append(feat(ou - h))
                for b in range(L):
                    Fn[b].append(feat(nat.blocks[b](h) - h))
        Fu = torch.cat(Fu)
        ck = [lin_cka(Fu, torch.cat(Fn[b])) for b in range(L)]
        cell = dict(i=i, k=k, nu_fisher=fitted(acc["fisher"]), nu_euclid=fitted(acc["euclid"]), nu_shiftR=fitted(acc["shiftR"]),
                    ka_copy_euclid=fitted(acc["euclid_copy"]), ka_copy_shiftR=fitted(acc["shiftR_copy"]),
                    nu_n125=fitted(snap[125]), nu_n250=fitted(snap[250]), half_energy=en / NA, kl_fwd=klf / NA, kl_rev=klr / NA,
                    cka=ck, cka_max=max(ck), cka_k=ck[k], cka_i=ck[i])
        res["cells"][k] = cell
        print(model, kind, p, k, {a: round(b, 4) for a, b in cell.items() if isinstance(b, float)}, flush=True)
    json.dump(res, open(f"{OUT}/p56_{model}_{kind}_{p}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
