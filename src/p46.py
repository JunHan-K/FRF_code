"""PREREG46: calibrate the out-of-frame threshold with trained (natural) blocks instead of synthetic ones.
Here every control is a trained local update with the same recipe (p14 upd: copy of B_u, 500 Adam steps, lr 1e-4, batch 32,
states of id_fit[:2000], MSE to a target block effect), so in-frame and out-of-frame controls share the approximation error:
  in-frame : upd_i (target d_i), upd_mid (target (d_i + d_j)/2), upd_j (target d_j)
  out      : upd_fo_r (target (1-r) d_j + r c_F (F(h) - h)), r = 0.5, 1; F = same-depth block of the foreign model (PREREG40/42
             pairs), c_F matches mean |c_F (F(h) - h)| to mean |d_j| on id_fit[:500]
  also     : base (B_u), copy_i, copy_j (known answers)
Per candidate on id_fit[:500] (measurement as p42: same input h, native suffix after k, midpoint Fisher p_m):
  FRC (mean of per-image ratios), off_sep = sqrt((Q - A^2/Dd) / Dd), off_share = (Q - A^2/Dd) / sum |e - e_skip|^2, and the
  training loss at the end (mean of the last 50 steps) relative to the target's mean square.
usage: p46.py <model> <perm> [comma-separated slots]"""
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

OUT = "out/p46"
NA, MB = 500, int(os.environ.get("P46_MB", 25))
UPD_STEPS = int(os.environ.get("P46_UPD_STEPS", 500))  # p14 recipe; smaller only for code tests
FOREIGN = {"deit_s": "deit3_s", "deit3_s": "augreg_s", "augreg_s": "deit_s",
           "deit_b": "deit3_b", "deit3_b": "augreg_b", "augreg_b": "deit_b"}


def fcov(a, b, p):
    return ((p * a * b).sum(-1) - (p * a).sum(-1) * (p * b).sum(-1)).double()


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

    def prefix(x, k):
        h = e13.emb(m, x)
        for b in m.blocks[:k]:
            h = b(h)
        return h

    def suf_nat(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, perm=p, foreign=FOREIGN[model], n_fit=NA, upd_steps=UPD_STEPS, cells={})
    for k in ks:
        i, j = D.KEEP[run][k], k
        Bi, Bj, Bu, Ff = nat.blocks[i], nat.blocks[j], m.blocks[k], fm.blocks[k]
        H = [prefix(xf[s:s + MB].to(S.dev), k).cpu() for s in range(0, NA, MB)]
        nj = sum(float(rownorm(Bj(h.to(S.dev)) - h.to(S.dev)).sum()) for h in H)
        nf = sum(float(rownorm(Ff(h.to(S.dev)) - h.to(S.dev)).sum()) for h in H)
        cF = nj / max(nf, 1e-30)
        targets = {"upd_i": lambda h: Bi(h) - h, "upd_mid": lambda h: 0.5 * (Bi(h) - h) + 0.5 * (Bj(h) - h),
                   "upd_j": lambda h: Bj(h) - h}
        for r in (0.5, 1.0):
            targets[f"upd_fo_{r}"] = lambda h, r=r: (1 - r) * (Bj(h) - h) + r * cF * (Ff(h) - h)
        Hk = torch.cat([prefix(xf[s:s + 100].to(S.dev), k).half().cpu() for s in range(0, NFIT, 100)])
        trained, tinfo = {}, {}
        for t_idx, (nm, tgt) in enumerate(targets.items()):
            g = torch.Generator().manual_seed(t_idx)  # own generator per control
            blk = copy.deepcopy(Bu)
            losses, ysq = [], []
            with torch.enable_grad():
                for q in blk.parameters():
                    q.requires_grad_(True)
                opt = torch.optim.Adam(blk.parameters(), lr=1e-4)
                for _ in range(UPD_STEPS):
                    hb = Hk[torch.randint(len(Hk), (32,), generator=g)].to(S.dev).float()
                    with torch.no_grad():
                        y = tgt(hb)
                    loss = ((blk(hb) - hb - y) ** 2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss))
                    ysq.append(float(y.pow(2).mean()))
            blk.eval()
            for q in blk.parameters():
                q.requires_grad_(False)
            trained[nm] = blk
            n50 = min(50, len(losses))
            tinfo[nm] = dict(init_loss=sum(losses[:n50]) / n50, final_loss=sum(losses[-n50:]) / n50,
                             rel_loss=sum(losses[-n50:]) / max(sum(ysq[-n50:]), 1e-30))
        del Hk
        C = {"copy_i": lambda h: Bi(h), "copy_j": lambda h: Bj(h), "base": lambda h: Bu(h)}
        for nm, blk in trained.items():
            C[nm] = lambda h, blk=blk: blk(h)
        acc = {nm: dict(R=[], A=0.0, Dd=0.0, Q=0.0, En=0.0) for nm in C}
        for h in H:
            h = h.to(S.dev)
            zs = suf_nat(h, k)
            zi, zj = suf_nat(Bi(h), k), suf_nat(Bj(h), k)
            pm = (0.5 * (zi + zj)).softmax(-1)
            den = fcov(zj - zi, zj - zi, pm)
            for nm, f in C.items():
                z = suf_nat(f(h), k)
                a = acc[nm]
                num = fcov(z - zi, zj - zi, pm)
                a["R"].append((num / den.clamp_min(1e-12)).cpu())
                a["A"] += float(num.sum())
                a["Dd"] += float(den.sum())
                a["Q"] += float(fcov(z - zi, z - zi, pm).sum())
                a["En"] += float(fcov(z - zs, z - zs, pm).sum())
        cell = dict(i=i, j=j, c_F=cF, cands={})
        for nm, a in acc.items():
            rr = max(a["Q"] - a["A"] ** 2 / max(a["Dd"], 1e-30), 0.0)
            cell["cands"][nm] = dict(FRC=float(torch.cat(a["R"]).mean()), FRC_pooled=a["A"] / max(a["Dd"], 1e-30),
                                     off_sep=(rr / max(a["Dd"], 1e-30)) ** 0.5, off_share=rr / max(a["En"], 1e-30),
                                     **tinfo.get(nm, {}))
        res["cells"][k] = cell
        print(model, p, k, {nm: (round(v["FRC"], 3), round(v["off_sep"], 3), round(v.get("rel_loss", 0), 3))
                            for nm, v in cell["cands"].items()}, flush=True)
    tag = f"_{slots.replace(',', '-')}" if slots else ""
    json.dump(res, open(f"{OUT}/p46_{model}_{p}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
