"""PREREG14: ten bridge measures between the local update and the behavioural effect, scored on known-answer controls.
All measures put the three blocks (measured B_k, native B_i, native B_j) on the SAME input h (prefix of the measured
model at slot k) and read them through a FIXED reference suffix: the native model after depth j (= k).
usage: p14.py <model> <perm> <slot>"""
import copy
import json
import math
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
from stitch import kl  # noqa: E402

OUT = "out/p14"
NA, MB = int(os.environ.get("P14_NA", 500)), int(os.environ.get("P14_MB", 25))  # P14_MB: smaller micro-batch for ViT-B
# PREREG25 options: measurement image set / offset / count, reference downstream, linearisation point, dose mode
XSET, XOFF = os.environ.get("P14_XSET", "id_hold"), int(os.environ.get("P14_XOFF", 0))
REF, LIN, MODE, TAG = (os.environ.get("P14_REF", "nat"), os.environ.get("P14_LIN", "mid"), os.environ.get("P14_MODE", ""),
                       os.environ.get("P14_TAG", ""))
TGRID = torch.linspace(-0.5, 1.5, 41)
LENS = os.environ.get("P14_LENS", "0").lower() not in ("", "0", "false", "no")  # PREREG33: add logit-lens (m12) and tuned-lens (m13) readouts of the slot output
SPEC = os.environ.get("P14_SPEC", "0").lower() not in ("", "0", "false", "no")  # PREREG34: role spectrum over the native blocks (m14)
KDREF = os.environ.get("P14_KDREF", "0").lower() not in ("", "0", "false", "no")  # PREREG37-1: distil the kd controls through the native reference suffix
NOM10 = os.environ.get("P14_NOM10", "0").lower() not in ("", "0", "false", "no")  # PREREG37: skip the edit-cost measure (m10)
EXACT = os.environ.get("P14_EXACT", "0").lower() not in ("", "0", "false", "no")  # PREREG37-3/4: FRA <-> spectrum ladder, save per-image Gram pieces
GRAM_KEEP = ("base", "copy", "upd0", "kd_pos", "scale_2.0")


def proj(a, b):
    """per-row scalar projection coefficient <a,b>/<b,b> on flattened rows"""
    a, b = a.flatten(1), b.flatten(1)
    return (a * b).sum(1) / b.pow(2).sum(1).clamp_min(1e-12)


def cen(z):
    return z - z.mean(-1, keepdim=True)  # logits are defined up to a constant


def merged(blk):
    """copy of blk with every LoRA edit folded into a plain Linear (so controls are ordinary blocks)"""
    blk = copy.deepcopy(blk)
    for parent, n in p11.lora_targets(blk):
        mod = getattr(parent, n)
        if isinstance(mod, p11.LoRA):
            lin = mod.lin
            with torch.no_grad():
                lin.weight += (mod.B @ mod.A).to(lin.weight.dtype)
            setattr(parent, n, lin)
    return blk


def main(model, p, slot, sol=None, tag=None):
    """sol: optional file of slots 6..11 block states (a refit solution, PREREG15); then only that solution's block
    at slot k is measured (no controls) and the measured model's slots 6..11 are replaced by the solution."""
    out = OUT if sol is None else OUT.replace("p14", "p15")
    os.makedirs(out, exist_ok=True)
    k = int(slot)
    run, pt, _ = p11.load_case(model, p)
    i, j = D.KEEP[run][k], k
    nat = P.nofuse(S.load(D.M).eval())
    for q in nat.parameters():
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    if sol is not None:
        for kk, st in torch.load(sol, map_location=S.dev).items():
            m.blocks[int(kk)].load_state_dict(st)
    for q in m.parameters():
        q.requires_grad_(False)
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, 2000, 100)])
    xh, _ = S.cache(XSET)
    xh = xh[XOFF:XOFF + NA]
    xh = torch.cat([nrm(xh[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, len(xh), 100)])
    Bi, Bj, base = nat.blocks[i], nat.blocks[j], m.blocks[k]
    g = torch.Generator().manual_seed(0)

    def prefix(net, x, upto):
        h = e13.emb(net, x)
        for b in net.blocks[:upto]:
            h = b(h)
        return h

    def suffix(o, start=None):
        if REF == "meas" and start is None:  # PREREG25 E: the measured model's own downstream instead of the native one
            return P.tail_fn(m, o, k + 1)
        for b in nat.blocks[(j + 1 if start is None else start):]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    with torch.no_grad():
        Hk = torch.cat([prefix(m, xf[s:s + 100].to(S.dev), k).half().cpu() for s in range(0, 2000, 100)])

    lens_info = {}
    if LENS:  # PREREG33: tuned-lens translator for native depth j (state after native block j -> native final logits)
        with torch.no_grad():
            Hn = torch.cat([prefix(nat, xf[s:s + 100].to(S.dev), j + 1).half().cpu() for s in range(0, 2000, 100)])
            Zn = torch.cat([nat(xf[s:s + 100].to(S.dev)).float().log_softmax(-1).cpu() for s in range(0, 2000, 100)])
        T = torch.nn.Linear(Hn.shape[-1], Hn.shape[-1]).to(S.dev)
        torch.nn.init.zeros_(T.weight)
        torch.nn.init.zeros_(T.bias)
        opt = torch.optim.Adam(T.parameters(), lr=1e-3)
        gl = torch.Generator().manual_seed(1)  # own generator: the control recipes keep using g
        for _ in range(500 if j + 1 < len(nat.blocks) else 0):  # last depth: logit lens is exact, translator = identity
            ix = torch.randint(0, 2000, (64,), generator=gl)
            hb, tb = Hn[ix].to(S.dev).float(), Zn[ix].to(S.dev)
            lq = nat.forward_head(nat.norm(hb + T(hb))).float().log_softmax(-1)
            loss = (tb.exp() * (tb - lq)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        for q in T.parameters():
            q.requires_grad_(False)
        with torch.no_grad():
            ll = lambda o: nat.forward_head(nat.norm(o)).float().log_softmax(-1)
            ev = lambda f: float(sum((Zn[s:s + 100].to(S.dev).exp() * (Zn[s:s + 100].to(S.dev) - f(Hn[s:s + 100].to(S.dev).float()))).sum(-1).sum()
                                     for s in range(0, 2000, 100)) / 2000)
            lens_info = dict(kl_logit_lens=ev(ll), kl_tuned_lens=ev(lambda o: ll(o + T(o))))
        del Hn, Zn

    # ---------------- controls (same recipes as PREREG9/11) ----------------
    def train_upd(lam):
        blk = copy.deepcopy(base)
        for q in blk.parameters():
            q.requires_grad_(True)
        opt = torch.optim.Adam(blk.parameters(), lr=1e-4)
        for it in range(500):
            h = Hk[torch.randint(len(Hk), (32,), generator=g)].to(S.dev).float()
            with torch.no_grad():
                y = (1 - lam) * (Bi(h) - h) + lam * (Bj(h) - h)
            loss = ((blk(h) - h - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return blk

    def train_kd(target, rank=None, steps=1000, start=None, ref_suffix=False):
        """output distillation of slot k toward the net with `target` at slot k. ref_suffix=False: measured model's own
        suffix (PREREG9 recipe); True: native reference suffix (used by measure 10)."""
        blk = copy.deepcopy(start if start is not None else base)
        if rank:
            for q in blk.parameters():
                q.requires_grad_(False)
            for parent, n in p11.lora_targets(blk):
                setattr(parent, n, p11.LoRA(getattr(parent, n), rank))
            ps = [q for n, q in blk.named_parameters() if n.endswith(".A") or n.endswith(".B")]
        else:
            for q in blk.parameters():
                q.requires_grad_(True)
            ps = list(blk.parameters())
        opt = torch.optim.Adam(ps, lr=1e-3 if rank else 1e-4)
        tail = (lambda o: suffix(o)) if ref_suffix else (lambda o: P.tail_fn(m, o, k + 1))
        for it in range(steps):
            h = Hk[torch.randint(len(Hk), (32,), generator=g)].to(S.dev).float()
            with torch.no_grad():
                zt = tail(target(h)).log_softmax(-1)
            zs = tail(blk(h)).log_softmax(-1)
            loss = (zt.exp() * (zt - zs)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        blk.eval()
        for q in blk.parameters():
            q.requires_grad_(False)
        return blk

    def train_kd_mix(lam, steps=1000):
        """PREREG25 A: output distillation toward the mixture (1 - lam) p_i + lam p_j (measured model's own suffix)"""
        blk = copy.deepcopy(base)
        for q in blk.parameters():
            q.requires_grad_(True)
        opt = torch.optim.Adam(blk.parameters(), lr=1e-4)
        tail = lambda o: P.tail_fn(m, o, k + 1)
        for it in range(steps):
            h = Hk[torch.randint(len(Hk), (32,), generator=g)].to(S.dev).float()
            with torch.no_grad():
                pt = (1 - lam) * tail(Bi(h)).softmax(-1) + lam * tail(Bj(h)).softmax(-1)
                zt = pt.clamp_min(1e-12).log()
            zs = tail(blk(h)).log_softmax(-1)
            loss = (pt * (zt - zs)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        blk.eval()
        for q in blk.parameters():
            q.requires_grad_(False)
        return blk

    if MODE == "dose":
        controls = {}
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            controls[f"upd_{lam}"] = (train_upd(lam), lam)
            controls[f"kdmix_{lam}"] = (train_kd_mix(lam), lam)
        controls["base"] = (base, None)
    else:
        controls = {"sol": (base, None)} if sol is not None else {"upd0": (train_upd(0.0), 0), "upd1": (train_upd(1.0), 1), "kd_neg": (train_kd(Bi, ref_suffix=KDREF), 0),
                    "kd_pos": (train_kd(Bj, ref_suffix=KDREF), 1), "kd_pos_lora1": (merged(train_kd(Bj, rank=1, ref_suffix=KDREF)), 1),
                    "copy": (copy.deepcopy(Bj), 1), "base": (base, None)}
    if SPEC and sol is None and MODE != "dose":  # PREREG34: strength controls h + gamma (B_j(h) - h), truth role j
        class Scaled(torch.nn.Module):
            def __init__(self, blk, gm):
                super().__init__()
                self.blk, self.gm = blk, gm

            def forward(self, x):
                return x + self.gm * (self.blk(x) - x)

        for gm in (0.5, 1.5, 2.0):
            controls[f"scale_{gm}"] = (Scaled(copy.deepcopy(Bj), gm), 1)
    if KDREF and sol is None and MODE != "dose":  # PREREG37-1 paired check: the old recipe (own suffix), trained last
        controls["kd_pos_own"] = (train_kd(Bj), 1)  # so the generator order of the standard controls is unchanged
    if SPEC:  # identifiability test: a near-duplicate of B_j added as an extra reference (weights + 1e-3 * std noise)
        Bdup = copy.deepcopy(Bj)
        gd = torch.Generator().manual_seed(2)
        with torch.no_grad():
            for q in Bdup.parameters():
                q += 1e-3 * q.std().clamp_min(1e-8) * torch.randn(q.shape, generator=gd).to(q.device, q.dtype)
        Bdup.eval()
    for b, _ in controls.values():
        b.eval()
        for q in b.parameters():
            q.requires_grad_(False)

    # ---------------- measures ----------------
    @torch.no_grad()
    def logp_m(blk):
        cur = m.blocks[k]
        m.blocks[k] = blk
        lp = torch.cat([m(xh[s:s + 100].to(S.dev)).float().log_softmax(-1) for s in range(0, NA, 100)])
        m.blocks[k] = cur
        return lp

    lp_mi, lp_mj = logp_m(Bi), logp_m(Bj)
    with torch.no_grad():
        ytop = torch.cat([nat(xh[s:s + 100].to(S.dev)).argmax(-1) for s in range(0, NA, 100)])

    def measures(blk, gram_name=None):
        acc = {n: [] for n in ("alpha", "m1_out_exact", "m2_out_jvp", "m3_fisher", "m4_simplex_t", "m6_cls",
                               "m7_native_ctx", "m9_backward", "m11_cfe") + (("m12_logit_lens", "m13_tuned_lens") if LENS else ())}
        prof = []
        if SPEC:  # PREREG34 role spectrum: per-image Fisher Gram pieces, analysed by spec.py after the loop
            nL = len(nat.blocks)
            sg = {"G": [], "c": [], "u": []}
            sd2 = {"G": [], "c": [], "u": []}  # references + near-duplicate of B_j
            lad = {"r1n": [], "r1d": []}
        dec = []
        for s in range(0, NA, MB):
            x = xh[s:s + MB].to(S.dev)
            with torch.no_grad():
                h = prefix(m, x, k)
                ou, oi, oj = blk(h), Bi(h), Bj(h)
                a, b = ou - oi, oj - oi
                acc["alpha"].append(proj(a, b).cpu())
                lay = getattr(m, "layout", "BNC")  # PREREG19: spatial mean instead of CLS for Swin/ConvNeXt
                pool = (lambda t: t[:, :1]) if lay == "BNC" else (lambda t: t.mean((1, 2))) if lay == "BHWC" else (lambda t: t.mean((2, 3)))
                acc["m6_cls"].append(proj(pool(a), pool(b)).cpu())
                zu, zi, zj = suffix(ou), suffix(oi), suffix(oj)
                acc["m1_out_exact"].append(proj(cen(zu) - cen(zi), cen(zj) - cen(zi)).cpu())
                # 11. PREREG28 CFE-alpha: exact logit effects, Fisher metric of the midpoint predictive distribution
                pm = (0.5 * (zi + zj)).softmax(-1)
                ea, eb = zu - zi, zj - zi
                fab_e = (pm * ea * eb).sum(-1) - (pm * ea).sum(-1) * (pm * eb).sum(-1)
                fbb_e = (pm * eb * eb).sum(-1) - (pm * eb).sum(-1).pow(2)
                acc["m11_cfe"].append((fab_e / fbb_e.clamp_min(1e-12)).cpu())
                if LENS:  # PREREG33: decode the slot output directly with the final norm + head (logit lens) or through
                    # the trained affine translator first (tuned lens); same centred-logit projection as m1
                    for nm, dz in (("m12_logit_lens", lambda o: nat.forward_head(nat.norm(o)).float()),
                                   ("m13_tuned_lens", lambda o: nat.forward_head(nat.norm(o + T(o))).float())):
                        lu, li, lj = dz(ou), dz(oi), dz(oj)
                        acc[nm].append(proj(cen(lu) - cen(li), cen(lj) - cen(li)).cpu())
                if SPEC:  # PREREG34 role spectrum: e_u = s * sum_b w_b e_b + r (Fisher metric of p_u, origin = block skipped)
                    z0 = cen(suffix(h))
                    E = torch.stack([cen(suffix(nat.blocks[bb](h))) - z0 for bb in range(nL)], 1)  # MB x L x C
                    eu = (cen(zu) - z0)[:, None]
                    pu_ = zu.softmax(-1)[:, None]
                    sg["G"].append(spec.fip(E, E, pu_).cpu())
                    sg["c"].append(spec.fip(E, eu, pu_)[..., 0].cpu())
                    sg["u"].append(spec.fip(eu, eu, pu_)[:, 0, 0].cpu())
                    if EXACT:  # PREREG37-3: R1 direct (p_u metric, origin B_i) and its pooled pieces
                        dI, dJ = E[:, i], E[:, j]
                        num = spec.fip((eu[:, 0] - dI)[:, None], (dJ - dI)[:, None], pu_)[:, 0, 0]
                        den = spec.fip((dJ - dI)[:, None], (dJ - dI)[:, None], pu_)[:, 0, 0]
                        lad["r1n"].append(num.cpu())
                        lad["r1d"].append(den.cpu())
                    if "copy" in controls and (blk is controls["copy"][0] or blk is controls["upd0"][0]):
                        E2 = torch.cat([E, (cen(suffix(Bdup(h))) - z0)[:, None]], 1)
                        sd2["G"].append(spec.fip(E2, E2, pu_).cpu())
                        sd2["c"].append(spec.fip(E2, eu, pu_)[..., 0].cpu())
                        sd2["u"].append(sg["u"][-1])
                # 4. simplex interpolation: t minimising KL(p_u || softmax((1-t) z_i + t z_j))
                lpu = zu.log_softmax(-1)
                kls = torch.stack([(lpu.exp() * (lpu - ((1 - t) * zi + t * zj).log_softmax(-1))).sum(-1) for t in TGRID])
                acc["m4_simplex_t"].append(TGRID[kls.argmin(0).cpu()])
                # 8. decision level
                ti, tj, tu = zi.argmax(-1), zj.argmax(-1), zu.argmax(-1)
                dis = ti != tj
                dec.append(torch.stack([((tu == tj) & dis).sum(), ((tu == ti) & dis).sum(), dis.sum()]).cpu())
                # 5. depth profile: propagate the three block outputs through native blocks j+1.. and project at each depth
                pu, pi_, pj = ou, oi, oj
                row = []
                for d in range(j + 1, len(nat.blocks)):
                    pu, pi_, pj = nat.blocks[d](pu), nat.blocks[d](pi_), nat.blocks[d](pj)
                    row.append(proj(pu - pi_, pj - pi_).mean().item())
                row.append(acc["m1_out_exact"][-1].mean().item())
                prof.append(row)
                # 7. native context: the native model's own state entering depth j
                hn = prefix(nat, x, j)
                acc["m7_native_ctx"].append(proj(blk(hn) - Bi(hn), Bj(hn) - Bi(hn)).cpu())
            # 2./3. first-order bridge at the midpoint state, plain and Fisher metric
            mid = (oi if LIN == "i" else 0.5 * (oi + oj)).detach()  # PREREG25 E: linearisation point
            z0, Ja = torch.func.jvp(suffix, (mid,), (a,))
            _, Jb = torch.func.jvp(suffix, (mid,), (b,))
            Ja, Jb = cen(Ja.detach()), cen(Jb.detach())
            acc["m2_out_jvp"].append(proj(Ja, Jb).cpu())
            pr = z0.detach().softmax(-1)
            fab = (pr * Ja * Jb).sum(-1) - (pr * Ja).sum(-1) * (pr * Jb).sum(-1)
            fbb = (pr * Jb * Jb).sum(-1) - (pr * Jb).sum(-1).pow(2)
            acc["m3_fisher"].append((fab / fbb.clamp_min(1e-12)).cpu())
            # 9. backward: credit routed to the block input. cotangent = grad of native top-1 log-prob through the
            # reference suffix at the midpoint state
            mid.requires_grad_(True)
            lpm = suffix(mid).log_softmax(-1).gather(1, ytop[s:s + MB, None]).sum()
            cot = torch.autograd.grad(lpm, mid)[0].detach()
            sv = {}
            for name, bb in (("u", blk), ("i", Bi), ("j", Bj)):
                hh = h.detach().clone().requires_grad_(True)
                sv[name] = torch.autograd.grad((bb(hh) * cot).sum(), hh)[0].detach()
            acc["m9_backward"].append(proj(sv["u"] - sv["i"], sv["j"] - sv["i"]).cpu())
        r = {n: float(torch.cat(v).float().mean()) for n, v in acc.items()}
        if SPEC:  # PREREG34 (amended): composition w, strength s, novelty nu, identifiability groups, per-image values
            Gx, cx, ux = torch.cat(sg["G"]), torch.cat(sg["c"]), torch.cat(sg["u"])
            A = spec.analyze(Gx, cx, ux, list(range(nL)))
            pi = A.pop("pi")
            gj = next(t for t, g in enumerate(A["groups"]) if j in g)
            gi = next(t for t, g in enumerate(A["groups"]) if i in g)
            r.update(m14_spec_wj=A["w"][j], m14_spec_wi=A["w"][i], m14_spec_s=A["s"], m14_spec_nov=A["nu"], m14_spec_nu_span=A["nu_span"],
                     m14_spec_gwj=A["group_w"][gj], m14_spec_gwi=A["group_w"][gi] if gi != gj else None,
                     m14_spec_ij_same_group=bool(gi == gj), m14_spec_cond=A["cond"], m14_spec=A)
            r["m14_pi_median"] = dict(wj=float(pi["w"][:, j].median()), wi=float(pi["w"][:, i].median()),
                                      s=float(pi["s"].median()), nu=float(pi["nu"].median()),
                                      nu_span=float(pi["nu_span"].median()))
            r["m14_pi"] = dict(wj=[round(float(x), 4) for x in pi["w"][:, j]], wi=[round(float(x), 4) for x in pi["w"][:, i]],
                               s=[round(float(x), 4) for x in pi["s"]], nu=[round(float(x), 4) for x in pi["nu"]],
                               nu_span=[round(float(x), 4) for x in pi["nu_span"]])
            if sd2["G"]:  # identifiability test: B_j and its near-duplicate must share a group (truth j for copy)
                D2 = spec.analyze(torch.cat(sd2["G"]), torch.cat(sd2["c"]), torch.cat(sd2["u"]), list(range(nL)) + ["j_dup"])
                D2.pop("pi")
                gd_ = next(t for t, g in enumerate(D2["groups"]) if j in g)
                r["m14_dup_test"] = dict(j_dup_same_group=bool("j_dup" in D2["groups"][gd_]), group_w_j=D2["group_w"][gd_],
                                         w_j=D2["w"][j], w_dup=D2["w"][-1], groups=D2["groups"], cond=D2["cond"])
            if EXACT:  # PREREG37-3 ladder (R0 = m11 mean of per-image ratios, p_m metric)
                r1n, r1d = torch.cat(lad["r1n"]), torch.cat(lad["r1d"])
                Gii, Gjj, Gij = Gx[:, i, i], Gx[:, j, j], Gx[:, i, j]
                r2n, r2d = cx[:, j] - cx[:, i] - Gij + Gii, Gii - 2 * Gij + Gjj  # closed form from skip-origin Gram
                R3 = []
                for t in range(len(Gx)):  # affine 2-reference least squares, solved as a Lagrange system
                    G2 = torch.tensor([[Gii[t], Gij[t]], [Gij[t], Gjj[t]]], dtype=torch.float64)
                    K = torch.zeros(3, 3, dtype=torch.float64)
                    K[:2, :2], K[:2, 2], K[2, :2] = G2, 1.0, 1.0
                    rhs = torch.tensor([cx[t, i], cx[t, j], 1.0], dtype=torch.float64)
                    R3.append(torch.linalg.solve(K, rhs)[1])
                R3 = torch.stack(R3)
                Gp, cp = Gx.sum(0), cx.sum(0)
                K = torch.zeros(3, 3, dtype=torch.float64)
                K[:2, :2] = Gp[[i, j]][:, [i, j]]
                K[:2, 2], K[2, :2] = 1.0, 1.0
                R3p = float(torch.linalg.solve(K, torch.stack([cp[i], cp[j], torch.tensor(1.0, dtype=torch.float64)]))[1])
                P2 = spec.fit(Gp[[i, j]][:, [i, j]], cp[[i, j]], ux.sum())
                ok = r1d > 1e-12
                r["p37_ladder"] = dict(
                    R0=r["m11_cfe"], R1=float((r1n[ok] / r1d[ok]).mean()), R2=float((r2n[ok] / r2d[ok]).mean()),
                    R3=float(R3[ok].mean()), R1p=float(r1n.sum() / r1d.sum()), R3p=R3p,
                    R4_wj=float(P2["w"][1]), R4_s=float(P2["s"]), R5=A["w"][j], R6=A["group_w"][gj],
                    max_abs_r1_r2=float((r1n[ok] / r1d[ok] - r2n[ok] / r2d[ok]).abs().max()),
                    max_abs_r1_r3=float((r1n[ok] / r1d[ok] - R3[ok]).abs().max()),
                    med_abs_r1_r2=float((r1n[ok] / r1d[ok] - r2n[ok] / r2d[ok]).abs().median()),
                    med_abs_r1_r3=float((r1n[ok] / r1d[ok] - R3[ok]).abs().median()), n_ok=int(ok.sum()))
                if gram_name in GRAM_KEEP:
                    os.makedirs(f"{OUT.replace('p14', 'p25')}/gram", exist_ok=True)
                    import numpy as np
                    np.savez(f"{OUT.replace('p14', 'p25')}/gram/{model}_{p}_{k}_{gram_name}.npz", G=Gx.float().numpy(),
                             c=cx.float().numpy(), u=ux.float().numpy(), i=i, j=j)
            if blk is base:  # known-answer tests of the estimator on this cell's reference geometry (exact truth)
                r["m14_synthetic"] = spec.synthetic(Gx, j)
        dd = torch.stack(dec).sum(0).float()
        r["m8_decision"] = float((dd[0] - dd[1]) / dd[2].clamp_min(1))
        r["m8_n_disagree"] = int(dd[2])
        pr_ = torch.tensor(prof).mean(0)
        r["m5_profile"] = pr_.tolist()
        r["m5_depth_area"] = float(pr_.mean())
        lp = logp_m(blk)
        r["mc"] = float(kl(lp, lp_mi)) - float(kl(lp, lp_mj))
        return r

    res = dict(model=model, perm=p, slot=k, i=int(i), j=int(j), sol=tag, controls={},
               opts=dict(xset=XSET, xoff=XOFF, na=NA, ref=REF, lin=LIN, mode=MODE), lens=lens_info)
    for name, (blk, truth) in controls.items():
        r = measures(blk, name)
        if EXACT:  # PREREG37-1 KR1: distillation fit to B_j through the reference suffix and the measured model's own
            with torch.no_grad():
                kr, ko = 0.0, 0.0
                for s in range(0, NA, 100):
                    h = prefix(m, xh[s:s + 100].to(S.dev), k)
                    ot, ob = Bj(h), blk(h)
                    lt, lb = suffix(ot).log_softmax(-1), suffix(ob).log_softmax(-1)
                    kr += float((lt.exp() * (lt - lb)).sum(-1).sum())
                    lt, lb = P.tail_fn(m, ot, k + 1).float().log_softmax(-1), P.tail_fn(m, ob, k + 1).float().log_softmax(-1)
                    ko += float((lt.exp() * (lt - lb)).sum(-1).sum())
            r["p37_kl_to_j"] = dict(ref=kr / NA, own=ko / NA)
        # 10. edit-cost distance: rank-1 edits (200 steps, reference suffix) toward the i-role and the j-role; the
        # role that is cheaper to reach (lower final KL) is the one the block is closer to
        fin = {}
        for tn, tgt in (() if name.startswith("scale_") or NOM10 else (("i", Bi), ("j", Bj))):  # no edit-cost for strength controls
            e = train_kd(tgt, rank=1, steps=200, start=blk, ref_suffix=True)
            e.eval()
            with torch.no_grad():
                tot = 0.0
                for s in range(0, NA, 100):
                    h = prefix(m, xh[s:s + 100].to(S.dev), k)
                    lt, ls = suffix(tgt(h)).log_softmax(-1), suffix(e(h)).log_softmax(-1)
                    tot += float((lt.exp() * (lt - ls)).sum(-1).sum())
            fin[tn] = tot / NA
        r["m10_edit_cost"] = math.log(max(fin["i"], 1e-8) / max(fin["j"], 1e-8)) if fin else None
        r["m10_kl"] = fin
        r["truth"] = truth
        res["controls"][name] = r
        print(model, p, k, name, {a: (round(v, 4) if isinstance(v, float) else v) for a, v in r.items() if a not in ("m5_profile", "m14_pi", "m14_spec", "m14_synthetic")}, flush=True)
    if TAG:  # PREREG25 runs go to their own folder
        out = OUT.replace("p14", "p25")
        os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/{model}_{p}_{k}{f'_{tag}' if tag else ''}{f'_{TAG}' if TAG else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
