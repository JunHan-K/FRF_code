"""PREREG43 (experiment C): where in the reference suffix does the local/output dissociation arise? No training.
Cells: base (recovered block B_u) of every out/p25/<model>_<perm>_<slot>_lens.json cell of this model/perm (PREREG36 cells).
Readouts as p14 (native reference suffix after depth k, same input h): alpha = projection of B_u(h) - B_i(h) on B_j(h) - B_i(h)
(flattened tokens), R(d) = same projection after native blocks k+1..d (m5_profile), m1 = centred-logit projection (m1_out_exact).
Locate (id_fit[:500]): dR(d) = R(d) - R(d-1), d = k+1..11 (head step excluded), d* = argmax, concentration = dR(d*) / sum max(dR, 0).
Intervene (id_hold[500:], the images not used for the PREREG36 cell labels): in the reference suffix, skip native block d
(main) or replace it by the recovered net's block d (secondary), for every d in k+1..11; read alpha and m1 through it.
Reproduction check: alpha and m1 of base on id_hold[:500] without intervention (must match out/p25 within 1e-3).
usage: p43.py <model> <perm>   (needs the P14_TAG=lens runs of src/p14.py for the slots to read)"""
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

OUT = "out/p43"
SW = "."
NA, NCLS, MB = 500, 500, int(os.environ.get("P43_MB", 25))


def proj(a, b):
    a, b = a.flatten(1).double(), b.flatten(1).double()
    return (a * b).sum(1) / b.pow(2).sum(1).clamp_min(1e-12)


def cen(z):
    return z - z.mean(-1, keepdim=True)


@torch.no_grad()
def main(model, p):
    os.makedirs(OUT, exist_ok=True)
    run, pt, _ = p11.load_case(model, p)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    for q in list(nat.parameters()) + list(m.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(nat.blocks)
    ks = sorted(int(f.split("_")[-2]) for f in glob.glob(f"{SW}/out/p25/{model}_{p}_*_lens.json"))
    if not ks:  # slots = those measured by the PREREG25 p14 runs: P14_TAG=lens python src/p14.py <model> <perm> <slot>
        sys.exit(f"no out/p25/{model}_{p}_<slot>_lens.json; run P14_TAG=lens python src/p14.py {model} {p} <slot> first")
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])
    xh, _ = S.cache("id_hold")
    NH = len(xh)
    xh = torch.cat([nrm(xh[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NH, 100)])

    def prefix(x, k):
        h = e13.emb(m, x)
        for b in m.blocks[:k]:
            h = b(h)
        return h

    def run_suffix(o, k, d=None, mode=None):
        for t in range(k + 1, L):
            if t == d and mode == "skip":
                continue
            o = (m.blocks[t] if (t == d and mode == "swap") else nat.blocks[t])(o)
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, perm=p, n_fit=NA, hold_range=[NCLS, NH], cells={})
    for k in ks:
        i, j = D.KEEP[run][k], k
        Bi, Bj, Bu = nat.blocks[i], nat.blocks[j], m.blocks[k]
        ds = list(range(k + 1, L))
        # ---- locate on id_fit ----
        prof = []
        for s in range(0, NA, MB):
            h = prefix(xf[s:s + MB].to(S.dev), k)
            pu, pi_, pj = Bu(h), Bi(h), Bj(h)
            row = [proj(pu - pi_, pj - pi_)]
            for d in ds:
                pu, pi_, pj = nat.blocks[d](pu), nat.blocks[d](pi_), nat.blocks[d](pj)
                row.append(proj(pu - pi_, pj - pi_))
            zu, zi, zj = (nat.forward_head(nat.norm(t)).float() for t in (pu, pi_, pj))
            row.append(proj(cen(zu) - cen(zi), cen(zj) - cen(zi)))
            prof.append(torch.stack(row, 1).cpu())
        R = torch.cat(prof).mean(0)  # [alpha, R(k+1..11), m1]
        dR = (R[1:len(ds) + 1] - R[:len(ds)]).tolist()
        dstar = ds[int(torch.tensor(dR).argmax())] if ds else None
        pos = sum(max(x, 0.0) for x in dR)
        conc = (max(dR) / pos if pos > 0 else 0.0) if ds else None  # no positive step -> no locus (0)

        # ---- reproduction check (id_hold[:500]) and interventions (id_hold[500:]) ----
        def readout(lo, hi, d=None, mode=None):
            al, m1 = [], []
            for s in range(lo, hi, MB):
                h = prefix(xh[s:min(s + MB, hi)].to(S.dev), k)
                ou, oi, oj = Bu(h), Bi(h), Bj(h)
                al.append(proj(ou - oi, oj - oi).cpu())
                zu, zi, zj = run_suffix(ou, k, d, mode), run_suffix(oi, k, d, mode), run_suffix(oj, k, d, mode)
                m1.append(proj(cen(zu) - cen(zi), cen(zj) - cen(zi)).cpu())
            return float(torch.cat(al).mean()), float(torch.cat(m1).mean())

        a_rep, m1_rep = readout(0, NCLS)
        a0, m10 = readout(NCLS, NH)
        skip, swap = {}, {}
        for d in ds:
            skip[d] = readout(NCLS, NH, d, "skip")[1]
            swap[d] = readout(NCLS, NH, d, "swap")[1]
        res["cells"][k] = dict(i=i, j=j, profile_fit=R.tolist(), dR=dR, dstar=dstar, concentration=conc,
                               repro=dict(alpha=a_rep, m1=m1_rep), hold=dict(alpha=a0, m1=m10, gap=m10 - a0),
                               m1_skip=skip, m1_swap=swap)
        print(model, p, k, "dstar", dstar, "conc", None if conc is None else round(conc, 3), "gap", round(m10 - a0, 4),
              {d: round(v - a0, 4) for d, v in skip.items()}, flush=True)
    json.dump(res, open(f"{OUT}/p43_{model}_{p}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
