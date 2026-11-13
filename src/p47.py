"""PREREG47 (E1): is the out-of-frame component of recovered blocks caused by relocation or by retraining alone?
Two kinds of KD-recovered networks, same recipe (depth/deit_perm.py: 3000 AdamW steps, all parameters, KD to the original):
  L : late half shuffled (D_La..D_Lf).   Cells: slots k in 6..11 with i = KEEP[k] != k   -> relocated, then retrained
  E : early half shuffled (D_Ea, D_Eb).  Cells: slots k in 6..11 (all in place, i = k)  -> retrained in place
Measurement as p42 (same code path, same precision): input h = m's own state entering k, native suffix after k, origin = skip,
metric p_candidate; nu_span12 = share of the effect outside the span of the 12 native block effects (spec.fit), id_fit[:500].
Candidates per cell:
  base       : m.blocks[k] (the retrained block)
  copy_k     : native B_k (known answer nu = 0);  copy_i : native B_i (L only, known answer nu = 0)
  foreign    : h + c_F (F(h) - h), F = same-depth block of the PREREG42 foreign model, c_F matches mean |.| to mean |B_k(h) - h|
               (known answer: large nu)
Also per cell: Fisher energy of the base effect, relative weight displacement |theta_u - theta_start| / |theta_start| with
start = native B_i (the block the slot held before retraining), and the network's final KL to the teacher (training json).
usage: p47.py <model> <L|E|N> <perm>[_seed1] [comma-separated slots]"""
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

OUT = os.environ.get("P47_OUT", "out/p47")  # PREREG51 writes trajectories elsewhere
NA, MB = 500, int(os.environ.get("P47_MB", 25))
FOREIGN = {"deit_s": "deit3_s", "deit3_s": "augreg_s", "augreg_s": "deit_s",
           "deit_b": "deit3_b", "deit3_b": "augreg_b", "augreg_b": "deit_b"}


def rownorm(t):
    return t.flatten(1).double().norm(dim=1)


def e_paths(model, p, kind="E"):
    """in-place checkpoints written by src/p47train.py (E: early-shuffled; N: noise control, PREREG49; U: unperturbed).
    p = 'a' | 'b' | ..., optionally with '_seed1' (seed replicate)."""
    q, _, sd = p.partition("_")
    sfx = f"_{sd}" if sd else ""
    base = f"out/s12r_{model}{sfx}"
    run = f"D_{kind}{q}"
    if kind in ("N", "U"):  # PREREG58: U = unperturbed network re-trained with the same recipe
        D.KEEP[run] = list(range(12))
    return run, f"{base}/{run}.pt", f"{base}/{run}.json"


@torch.no_grad()
def main(model, kind, p, slots=None):
    os.makedirs(OUT, exist_ok=True)
    if kind == "L":
        run, pt, _ = p11.load_case(model, p)
        rj = e13.paths(model, run)[2]
    else:
        D.M = s12r.MODELS[model]
        run, pt, rj = e_paths(model, p, kind)
    if os.environ.get("P47_PT"):  # PREREG51: an intermediate checkpoint of a trajectory re-run
        pt = os.environ["P47_PT"]
        rj = os.environ.get("P47_RJ", rj)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    fm = S.load(s12r.MODELS[FOREIGN[model]]).eval()
    for q in list(nat.parameters()) + list(m.parameters()) + list(fm.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(m.blocks)
    ks = [int(t) for t in slots.split(",")] if slots else \
        [k for k in range(6, L) if (D.KEEP[run][k] != k if kind == "L" else D.KEEP[run][k] == k)]
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])
    kl_final = json.load(open(rj))["ckpt"][-1]["kl_teacher"]

    def prefix(x, k):
        h = e13.emb(m, x)
        for b in m.blocks[:k]:
            h = b(h)
        return h

    def suf_nat(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, kind=kind, perm=p, run=run, foreign=FOREIGN[model], n_fit=NA, kl_final=kl_final, cells={})
    for k in ks:
        i = D.KEEP[run][k]
        Bi, Bk, Bu, Ff = nat.blocks[i], nat.blocks[k], m.blocks[k], fm.blocks[k]
        H = [prefix(xf[s:s + MB].to(S.dev), k).cpu() for s in range(0, NA, MB)]
        nk = sum(float(rownorm(Bk(h.to(S.dev)) - h.to(S.dev)).sum()) for h in H)
        nf = sum(float(rownorm(Ff(h.to(S.dev)) - h.to(S.dev)).sum()) for h in H)
        cF = nk / max(nf, 1e-30)
        C = {"base": lambda h: Bu(h), "copy_k": lambda h: Bk(h), "foreign": lambda h: h + cF * (Ff(h) - h)}
        if i != k:
            C["copy_i"] = lambda h: Bi(h)
        acc = {nm: dict(G=torch.zeros(L, L, dtype=torch.float64), c=torch.zeros(L, dtype=torch.float64), u=0.0) for nm in C}
        for h in H:
            h = h.to(S.dev)
            zs = suf_nat(h, k)
            E = torch.stack([suf_nat(nat.blocks[b](h), k) - zs for b in range(L)], 1)
            for nm, f in C.items():
                z = suf_nat(f(h), k)
                a = acc[nm]
                pz = z.softmax(-1)[:, None]
                e = (z - zs)[:, None]
                a["G"] += spec.fip(E, E, pz).sum(0).cpu()
                a["c"] += spec.fip(E, e, pz)[..., 0].sum(0).cpu()
                a["u"] += float(spec.fip(e, e, pz)[:, 0, 0].sum())
        su, s0 = Bu.state_dict(), Bi.state_dict()
        num = sum(float((su[n].double() - s0[n].double()).pow(2).sum()) for n in s0)
        den = sum(float(s0[n].double().pow(2).sum()) for n in s0)
        cell = dict(i=i, k=k, c_F=cF, w_disp=(num / den) ** 0.5, cands={})
        for nm, a in acc.items():
            Fit = spec.fit(a["G"], a["c"], torch.tensor(a["u"], dtype=torch.float64))
            cell["cands"][nm] = dict(nu_span12=float(Fit["nu_span"]), energy=a["u"] / NA)
        res["cells"][k] = cell
        print(model, kind, p, k, i, round(cell["w_disp"], 4),
              {nm: round(v["nu_span12"], 4) for nm, v in cell["cands"].items()}, flush=True)
    tag = (f"_{slots.replace(',', '-')}" if slots else "") + os.environ.get("P47_TAG", "")
    json.dump(res, open(f"{OUT}/p47_{model}_{kind}_{p}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
