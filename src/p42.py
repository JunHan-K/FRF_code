"""PREREG42 (experiment B): out-of-frame candidates. Does the adequacy term (off_sep) withhold a misleading coordinate?
Recovered model m (perm run), slot k with i != j (PREREG39 cells). Measurement as p14: same input h (m's state entering k),
native reference suffix after depth k, midpoint Fisher p_m = softmax((z_i + z_j) / 2).
Candidates (hidden space, d_b = B_b(h) - h):
  in-frame : copy_i, copy_j, interp_l = h + (1-l) d_i + l d_j (l = .25 .5 .75), scale_g = h + g d_j (g = .5, 2)
  out      : mix_<F>_r = h + (1-r) d_j + r c_F (F(h) - h), r = .25 .5 .75 1, F in {foreign (same depth, independent model of
             the same width, PREREG40 pairs), third (native B_m, m not in {i,j}, smallest max |Fisher cos| with e_i, e_j)};
             c_F matches mean |c_F (F(h) - h)| to mean |d_j| on id_fit[:500]
  natural  : base (= B_u), upd1 (local update trained toward d_j, p14 recipe, own generator)
Per candidate on id_fit[:500]:
  FRC (m11: mean of per-image ratios), pooled A = sum <e-e_i, e_j-e_i>, Dd = sum |e_j-e_i|^2, Q = sum |e-e_i|^2 (metric p_m),
  off_sep = sqrt((Q - A^2/Dd) / Dd), off_share = (Q - A^2/Dd) / sum |e - e_skip|^2 (metric p_m),
  nu_span over the 12 native blocks (metric p_candidate, origin skip; spec.fit), construction check (top-1 disagreement in m).
On id_hold (all images): agree_i / agree_j = top-1 agreement of the candidate net with the B_i- / B_j-net (m's suffix).
usage: p42.py <model> <perm> [comma-separated slots]"""
import copy
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

OUT = "out/p42"
NA, MB = 500, int(os.environ.get("P42_MB", 25))
UPD_STEPS = int(os.environ.get("P42_UPD_STEPS", 500))  # p14 recipe; smaller only for code tests
FOREIGN = {"deit_s": "deit3_s", "deit3_s": "augreg_s", "augreg_s": "deit_s",
           "deit_b": "deit3_b", "deit3_b": "augreg_b", "augreg_b": "deit_b", "clip_b": "deit_b"}
RHOS = (0.25, 0.5, 0.75, 1.0)


def fcov(a, b, p):
    return ((p * a * b).sum(-1) - (p * a).sum(-1) * (p * b).sum(-1)).double()


def cen(z):
    return z - z.mean(-1, keepdim=True)


def rownorm(t):
    return t.flatten(1).double().norm(dim=1)


@torch.no_grad()
def main(model, p, slots=None):
    os.makedirs(OUT, exist_ok=True)
    run, pt, _ = p11.load_case(model, p)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    fm = S.load(s12r.MODELS[FOREIGN[model]]).eval()
    for q in list(nat.parameters()) + list(m.parameters()) + list(fm.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(m.blocks)
    ks = [int(t) for t in slots.split(",")] if slots else [k for k in range(L) if D.KEEP[run][k] != k]
    xf, _ = S.cache("id_fit")
    NFIT = min(2000, len(xf))
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NFIT, 100)])
    xh, _ = S.cache("id_hold")
    NH = len(xh)
    xh = torch.cat([nrm(xh[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NH, 100)])

    def prefix(x, k):
        h = e13.emb(m, x)
        for b in m.blocks[:k]:
            h = b(h)
        return h

    def suf_nat(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    def suf_meas(o, k):
        return P.tail_fn(m, o, k + 1)

    res = dict(model=model, perm=p, foreign=FOREIGN[model], n_fit=NA, n_hold=NH, cells={})
    for k in ks:
        i, j = D.KEEP[run][k], k
        Bi, Bj, Bu, Ff = nat.blocks[i], nat.blocks[j], m.blocks[k], fm.blocks[k]
        H = [prefix(xf[s:s + MB].to(S.dev), k).cpu() for s in range(0, NA, MB)]
        # ---- pass 1: native effects (third-block choice) and norms for c_F ----
        Gn = torch.zeros(L, L, dtype=torch.float64)
        nj, nf = 0.0, {"foreign": 0.0}
        nn_ = torch.zeros(L, dtype=torch.float64)
        for h in H:
            h = h.to(S.dev)
            zs = suf_nat(h, k)
            zi, zj = suf_nat(Bi(h), k), suf_nat(Bj(h), k)
            pm = (0.5 * (zi + zj)).softmax(-1)[:, None]
            E = torch.stack([suf_nat(nat.blocks[b](h), k) - zs for b in range(L)], 1)
            Gn += spec.fip(E, E, pm).sum(0).cpu()
            nj += float(rownorm(Bj(h) - h).sum())
            nf["foreign"] += float(rownorm(Ff(h) - h).sum())
            for b in range(L):
                nn_[b] += float(rownorm(nat.blocks[b](h) - h).sum())
        dg = Gn.diagonal().clamp_min(1e-30).sqrt()
        cos = Gn / dg[:, None] / dg[None, :]
        others = [b for b in range(L) if b not in (i, j)]
        third = min(others, key=lambda b: max(abs(float(cos[b, i])), abs(float(cos[b, j]))))
        Ft = nat.blocks[third]
        nf["third"] = float(nn_[third])
        cF = {"foreign": nj / max(nf["foreign"], 1e-30), "third": nj / max(nf["third"], 1e-30)}
        Fs = {"foreign": Ff, "third": Ft}
        # ---- natural control upd1 (p14 recipe: 500 Adam steps, lr 1e-4, batch 32, states of id_fit[:2000]) ----
        Hk = torch.cat([prefix(xf[s:s + 100].to(S.dev), k).half().cpu() for s in range(0, NFIT, 100)])
        g = torch.Generator().manual_seed(0)
        upd1 = copy.deepcopy(Bu)
        with torch.enable_grad():
            for q in upd1.parameters():
                q.requires_grad_(True)
            opt = torch.optim.Adam(upd1.parameters(), lr=1e-4)
            for _ in range(UPD_STEPS):
                hb = Hk[torch.randint(len(Hk), (32,), generator=g)].to(S.dev).float()
                with torch.no_grad():
                    y = Bj(hb) - hb
                loss = ((upd1(hb) - hb - y) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        upd1.eval()
        for q in upd1.parameters():
            q.requires_grad_(False)
        del Hk
        # ---- candidates ----
        C = {"copy_i": lambda h: Bi(h), "copy_j": lambda h: Bj(h)}
        for lam in (0.25, 0.5, 0.75):
            C[f"interp_{lam}"] = lambda h, lam=lam: h + (1 - lam) * (Bi(h) - h) + lam * (Bj(h) - h)
        for gm in (0.5, 2.0):
            C[f"scale_{gm}"] = lambda h, gm=gm: h + gm * (Bj(h) - h)
        for fn, F in Fs.items():
            for r in RHOS:
                C[f"mix_{fn}_{r}"] = lambda h, F=F, r=r, c=cF[fn]: h + (1 - r) * (Bj(h) - h) + r * c * (F(h) - h)
        C["base"] = lambda h: Bu(h)
        C["upd1"] = lambda h: upd1(h)
        acc = {nm: dict(R=[], A=0.0, Dd=0.0, Q=0.0, En=0.0, G=torch.zeros(L, L, dtype=torch.float64),
                        c=torch.zeros(L, dtype=torch.float64), u=0.0, dis_i=0, dis_j=0) for nm in C}
        dis_ij = 0
        for h in H:
            h = h.to(S.dev)
            zs = suf_nat(h, k)
            zi, zj = suf_nat(Bi(h), k), suf_nat(Bj(h), k)
            pm = (0.5 * (zi + zj)).softmax(-1)
            E = torch.stack([suf_nat(nat.blocks[b](h), k) - zs for b in range(L)], 1)
            ti, tj = suf_meas(Bi(h), k).argmax(-1), suf_meas(Bj(h), k).argmax(-1)
            dis_ij += int((ti != tj).sum())
            for nm, f in C.items():
                o = f(h)
                z = suf_nat(o, k)
                a = acc[nm]
                num, den = fcov(z - zi, zj - zi, pm), fcov(zj - zi, zj - zi, pm)
                a["R"].append((num / den.clamp_min(1e-12)).cpu())
                a["A"] += float(num.sum())
                a["Dd"] += float(den.sum())
                a["Q"] += float(fcov(z - zi, z - zi, pm).sum())
                a["En"] += float(fcov(z - zs, z - zs, pm).sum())
                pz = z.softmax(-1)[:, None]
                e = (z - zs)[:, None]
                a["G"] += spec.fip(E, E, pz).sum(0).cpu()
                a["c"] += spec.fip(E, e, pz)[..., 0].sum(0).cpu()
                a["u"] += float(spec.fip(e, e, pz)[:, 0, 0].sum())
                tf = suf_meas(o, k).argmax(-1)
                a["dis_i"] += int((tf != ti).sum())
                a["dis_j"] += int((tf != tj).sum())
        cell = dict(i=i, j=j, third=third, c_F=cF, dis_ij_fit=dis_ij / NA, cands={})
        for nm, a in acc.items():
            rr = max(a["Q"] - a["A"] ** 2 / max(a["Dd"], 1e-30), 0.0)
            Fit = spec.fit(a["G"], a["c"], torch.tensor(a["u"], dtype=torch.float64))
            cell["cands"][nm] = dict(FRC=float(torch.cat(a["R"]).mean()), FRC_pooled=a["A"] / max(a["Dd"], 1e-30),
                                     off_sep=(rr / max(a["Dd"], 1e-30)) ** 0.5, off_share=rr / max(a["En"], 1e-30),
                                     nu_span12=float(Fit["nu_span"]), dis_i_fit=a["dis_i"] / NA, dis_j_fit=a["dis_j"] / NA)
        # ---- truth on id_hold ----
        ag = {nm: [0, 0] for nm in C}
        for s in range(0, NH, MB):
            h = prefix(xh[s:s + MB].to(S.dev), k)
            ti, tj = suf_meas(Bi(h), k).argmax(-1), suf_meas(Bj(h), k).argmax(-1)
            for nm, f in C.items():
                tf = suf_meas(f(h), k).argmax(-1)
                ag[nm][0] += int((tf == ti).sum())
                ag[nm][1] += int((tf == tj).sum())
        for nm in C:
            cell["cands"][nm].update(agree_i=ag[nm][0] / NH, agree_j=ag[nm][1] / NH)
        res["cells"][k] = cell
        print(model, p, k, "third", third, {nm: (round(v["FRC"], 3), round(v["off_sep"], 3), round(v["nu_span12"], 3),
                                                 round(v["agree_i"], 3), round(v["agree_j"], 3)) for nm, v in cell["cands"].items()}, flush=True)
    tag = f"_{slots.replace(',', '-')}" if slots else ""
    json.dump(res, open(f"{OUT}/p42_{model}_{p}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
