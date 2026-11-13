"""PREREG39: directional replacement in recovered shuffled ViTs (no training).
Recovered model m (perm run), slot k: recovered block B_u = m.blocks[k], source block B_i (i = KEEP[run][k]), slot owner
B_j (native block j = k). Input h = m's own state entering slot k.
Predictors (id_fit[:500], calibration):
  R_nat  : FRA coordinate (= p14 m11): per image <e_u - e_i, e_j - e_i>_F / |e_j - e_i|^2_F, Fisher at the midpoint
           p_m = softmax((z_i + z_j) / 2), z = native reference suffix after depth j; mean over images
  R_meas : same through m's own suffix (the network in which the replacement is made)
  R_lens : centred-logit projection of the slot outputs read by the native final norm + head (= p14 m12)
  alpha  : local projection of B_u(h) - B_i(h) on B_j(h) - B_i(h) (= p14 alpha)
  cka_diff = CKA(B_u, B_j) - CKA(B_u, B_i) on the CLS outputs;  kl_diff = KL(p_u || p_i) - KL(p_u || p_j) (m's suffix)
  direct baselines: agree_fit_diff = agree(B_j) - agree(B_i) on the calibration images (m's suffix), kl_diff_nat
  magnitude-only (u alone): fisher energy of e_u - e_skip, skip KL, |e_u|_F (cannot tell i from j; reported)
  separation = |e_j - e_i|^2_F / (|e_i - e_skip|^2_F + |e_j - e_skip|^2_F) (inclusion), all pooled, native suffix
Truth (id_hold, all 1976 images): replace B_u in m by every native block b (incl. i, j) as h + gamma (B_b(h) - h)
  fixed   : gamma = 1
  matched : gamma_b chosen on id_fit[:500] so that mean |gamma (B_b(h) - h)| = mean |B_u(h) - h| (local norm only)
  top-1 agreement with the net containing B_u (main: Dagree = agree(B_j) - agree(B_i)), KL(p_u || p_replaced), accuracy
  (D = top1(B_j) - top1(B_i) does not depend on B_u: reported as a slot property only).
Known-answer cells: B_u := copy of B_j (R = 1, D > 0) and B_u := copy of B_i (R = 0, D < 0), same slot.
usage: p39.py <model> <perm> [comma-separated slots]   (default: every slot with i != j)"""
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

OUT = "out/p39"
NA, MB = 500, int(os.environ.get("P39_MB", 25))


def cen(z):
    return z - z.mean(-1, keepdim=True)


def fcov(a, b, p):
    return ((p * a * b).sum(-1) - (p * a).sum(-1) * (p * b).sum(-1)).double()


def proj(a, b):
    a, b = a.flatten(1).double(), b.flatten(1).double()
    return (a * b).sum(1) / (b * b).sum(1).clamp_min(1e-12)


def lin_cka(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    return float((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm()))


def kl(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)


@torch.no_grad()
def main(model, p, slots=None):
    os.makedirs(OUT, exist_ok=True)
    run, pt, _ = p11.load_case(model, p)
    nat = P.nofuse(S.load(D.M).eval())
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    for q in list(nat.parameters()) + list(m.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(m.blocks)
    ks = [int(t) for t in slots.split(",")] if slots else [k for k in range(L) if D.KEEP[run][k] != k]
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])
    xh, yh = S.cache("id_hold")
    NH = len(xh)
    xh = torch.cat([nrm(xh[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NH, 100)])
    yh = yh.to(S.dev)

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

    def lens(o):
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, perm=p, n_fit=NA, n_hold=NH, cells={})
    for k in ks:
        i, j = D.KEEP[run][k], k
        Bi, Bj = nat.blocks[i], nat.blocks[j]
        cases = {"base": m.blocks[k], "ka_copy_j": copy.deepcopy(Bj), "ka_copy_i": copy.deepcopy(Bi)}
        for cname, Bu in cases.items():
            Bu.eval()
            # ---------------- predictors on id_fit ----------------
            acc = {t: [] for t in ("R_nat", "R_meas", "R_lens", "alpha")}
            pim = {t: [] for t in ("agree_i", "agree_j", "kl_i_meas", "kl_j_meas", "kl_i_nat", "kl_j_nat")}  # per image
            pool = dict(sep_n=0.0, sep_d=0.0, energy=0.0, enorm=0.0, skipkl=0.0, kl_ui=0.0, kl_uj=0.0)
            cu, ci, cj = [], [], []
            dnorm = {b: 0.0 for b in range(L)}
            unorm = 0.0
            for s in range(0, NA, MB):
                h = prefix(xf[s:s + MB].to(S.dev), k)
                ou, oi, oj = Bu(h), Bi(h), Bj(h)
                acc["alpha"].append(proj(ou - oi, oj - oi).cpu())
                for nm, sf in (("R_nat", lambda o: suf_nat(o, k)), ("R_meas", lambda o: suf_meas(o, k))):
                    zu, zi, zj = sf(ou), sf(oi), sf(oj)
                    pm = (0.5 * (zi + zj)).softmax(-1)
                    acc[nm].append((fcov(zu - zi, zj - zi, pm) / fcov(zj - zi, zj - zi, pm).clamp_min(1e-12)).cpu())
                    lu_, li2, lj2 = zu.log_softmax(-1), zi.log_softmax(-1), zj.log_softmax(-1)
                    tg = "nat" if nm == "R_nat" else "meas"
                    pim[f"kl_i_{tg}"].append(kl(lu_, li2).cpu())
                    pim[f"kl_j_{tg}"].append(kl(lu_, lj2).cpu())
                    if nm == "R_meas":  # direct replacement test on the calibration images (recovered net's own suffix)
                        pim["agree_i"].append((zi.argmax(-1) == zu.argmax(-1)).float().cpu())
                        pim["agree_j"].append((zj.argmax(-1) == zu.argmax(-1)).float().cpu())
                    if nm == "R_nat":
                        zs = sf(h)
                        pu = zu.softmax(-1)
                        pool["sep_n"] += float(fcov(zj - zi, zj - zi, pm).sum())
                        pool["sep_d"] += float((fcov(zi - zs, zi - zs, pu) + fcov(zj - zs, zj - zs, pu)).sum())
                        pool["energy"] += float(fcov(zu - zs, zu - zs, pu).sum())
                    else:
                        lu, li_, lj_ = zu.log_softmax(-1), zi.log_softmax(-1), zj.log_softmax(-1)
                        pool["kl_ui"] += float(kl(lu, li_).sum())
                        pool["kl_uj"] += float(kl(lu, lj_).sum())
                        pool["skipkl"] += float(kl(lu, sf(h).log_softmax(-1)).sum())
                pool["enorm"] += float((ou - h).flatten(1).double().norm(dim=1).sum())
                lu, li_, lj_ = lens(ou), lens(oi), lens(oj)
                acc["R_lens"].append(proj(cen(lu) - cen(li_), cen(lj_) - cen(li_)).cpu())
                cu.append(ou[:, 0].double().cpu())
                ci.append(oi[:, 0].double().cpu())
                cj.append(oj[:, 0].double().cpu())
                unorm += float((ou - h).flatten(1).double().norm(dim=1).sum())
                for b in range(L):
                    dnorm[b] += float((nat.blocks[b](h) - h).flatten(1).double().norm(dim=1).sum())
            Cu, Ci, Cj = torch.cat(cu), torch.cat(ci), torch.cat(cj)
            pred = {t: float(torch.cat(v).mean()) for t, v in acc.items()}
            P_ = {t: torch.cat(v) for t, v in pim.items()}
            pred.update(agree_fit_diff=float(P_["agree_j"].mean() - P_["agree_i"].mean()),
                        kl_diff_nat=float((P_["kl_i_nat"] - P_["kl_j_nat"]).mean()))
            per_image = None
            if cname == "base":  # PREREG39: sample-efficiency comparison (first n images of id_fit)
                per_image = {t: [round(float(x), 5) for x in torch.cat(v)] for t, v in acc.items()}
                per_image.update({t: [round(float(x), 5) for x in v] for t, v in P_.items()})
            pred.update(cka_diff=lin_cka(Cu, Cj) - lin_cka(Cu, Ci), kl_diff=(pool["kl_ui"] - pool["kl_uj"]) / NA,
                        energy=pool["energy"] / NA, skip_kl=pool["skipkl"] / NA, local_norm=pool["enorm"] / NA,
                        separation=pool["sep_n"] / max(pool["sep_d"], 1e-30))
            gam = {b: unorm / max(dnorm[b], 1e-30) for b in range(L)}
            # ---------------- truth on id_hold ----------------
            t_m = 0
            cor = {b: {"fixed": 0, "matched": 0} for b in range(L)}
            agr = {b: {"fixed": 0, "matched": 0} for b in range(L)}  # top-1 agreement with the net containing B_u
            klv = {b: {"fixed": 0.0, "matched": 0.0} for b in range(L)}
            for s in range(0, NH, MB):
                h = prefix(xh[s:s + MB].to(S.dev), k)
                y = yh[s:s + MB]
                zm = suf_meas(Bu(h), k)
                lpm = zm.log_softmax(-1)
                t_m += int((zm.argmax(-1) == y).sum())
                for b in range(L):
                    d = nat.blocks[b](h) - h
                    for cond, g in (("fixed", 1.0), ("matched", gam[b])):
                        z = suf_meas(h + g * d, k)
                        cor[b][cond] += int((z.argmax(-1) == y).sum())
                        agr[b][cond] += int((z.argmax(-1) == zm.argmax(-1)).sum())
                        klv[b][cond] += float(kl(lpm, z.log_softmax(-1)).sum())
            truth = dict(top1_u=t_m / NH, top1={b: {c: v / NH for c, v in d.items()} for b, d in cor.items()},
                         kl={b: {c: v / NH for c, v in d.items()} for b, d in klv.items()}, gamma_matched=gam)
            truth["agree"] = {b: {c: v / NH for c, v in d.items()} for b, d in agr.items()}
            truth["Dagree_fixed"] = truth["agree"][j]["fixed"] - truth["agree"][i]["fixed"]  # main truth (PREREG39)
            truth["Dagree_matched"] = truth["agree"][j]["matched"] - truth["agree"][i]["matched"]
            truth["D_fixed"] = truth["top1"][j]["fixed"] - truth["top1"][i]["fixed"]  # slot property (independent of B_u)
            truth["D_matched"] = truth["top1"][j]["matched"] - truth["top1"][i]["matched"]
            truth["Dkl_fixed"] = truth["kl"][i]["fixed"] - truth["kl"][j]["fixed"]  # > 0: B_j closer to the recovered net
            res["cells"].setdefault(k, {})[cname] = dict(i=i, j=j, pred=pred, truth=truth, per_image=per_image)
            print(model, p, k, cname, {t: round(v, 3) for t, v in pred.items()}, "Dagree", round(truth["Dagree_fixed"], 4), round(truth["Dagree_matched"], 4), "D", round(truth["D_fixed"], 4),
                  round(truth["D_matched"], 4), "top1_u", truth["top1_u"], flush=True)
    tag = f"_{slots.replace(',', '-')}" if slots else ""
    json.dump(res, open(f"{OUT}/p39_{model}_{p}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
