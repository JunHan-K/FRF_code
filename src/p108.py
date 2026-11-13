"""PREREG108/109 (GPU): one pipeline for several fine-grained forget groups and recent unlearning baselines.
Groups (ImageNet WordNet ranges): dogs 151-268 (118), birds 7-24 + 80-100 + 127-146 (59), reptiles 33-68 (36), insects 300-326 (27).
Arms:
  native    no unlearning (reference for probe and attacks)
  HRr       forget head rows re-initialised + dog-free repair (PREREG102)
  SD3-12H, SD10-12H   selective dampening on all blocks + head re-init + repair (PREREG102); OURS = PREREG102 selection rule
  RL        random non-forget labels on forget images + retain (PREREG104)
  DELETE    masked-teacher distillation on forget images (Zhou et al., CVPR 2025): KL(masked p_orig || p_student), all params
  Spotter   0.7 DELETE + 0.3 the same on PGD neighbours of forget images (eps 0.03, step 0.01, 3 its, against the original) +
            1.0 intra-class cosine dispersion of forget pre-logit features (ICML 2026, defence setting lambda2 = 1)
  SCRUB     teacher-student (Kurmanji et al., NeurIPS 2023): first 1,000 steps add a max-step (maximise KL(teacher || student) on a
            forget batch); every step a min-step on retain (KL to teacher + CE)
  RL / DELETE / Spotter / SCRUB: all parameters, AdamW lr 1e-5, no decay, batch 64, 3,000 steps, warmup + cosine, augmentation.
Measured (fresh set unless noted): final forget / retain, id_hold forget / retain (selection), linear probe (PREREG104, group-way),
LoRA attacks few1 / few5 / full_lora (PREREG104), strong all-parameter attacks sft1 / sft5 / sftall (PREREG106), prototypical
relearning attack PRA (Spotter paper): cosine prototypes of 5 forget id_fit images per class in pre-logit space replace the forget
head rows by interpolation alpha in {0, 0.1, ..., 1}, the largest alpha whose id_hold retain drop is <= 1 point is used.
Written 2026-11-08 08:46 KST, before any PREREG108/109 network exists.
usage: p108.py <group> <model> <arm>"""
import copy
import json
import math
import os
import sys

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import p96  # noqa: E402

GROUPS = {"dogs": list(range(151, 269)), "birds": list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),
          "reptiles": list(range(33, 69)), "insects": list(range(300, 327))}
if __name__ == "__main__":
    p96.FORGET = torch.tensor(GROUPS[sys.argv[1]])
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p91 import acc  # noqa: E402
from p99 import ce_attack, wrap, fold  # noqa: E402
from p100 import fisher, dampen  # noqa: E402
from p106 import strong_attacks  # noqa: E402
from spotter_sampling import spotter_classes, spotter_images  # noqa: E402

OUT = os.environ.get("P108_OUT", "out/p108")
STEPS = int(os.environ.get("P108_STEPS", 3000))  # paper: default value (override only for quick tests)
ATT = int(os.environ.get("P108_ATT", 300))


def FG():
    return p96.FORGET


def masked_logp(z):
    z = z.float().clone()
    z[:, FG().to(z.device)] = -1e4
    return z.log_softmax(-1)


def head_reinit(m):
    with torch.no_grad():
        w = m.head.weight
        w[FG().to(w.device)] = torch.randn(len(FG()), w.shape[1], generator=torch.Generator().manual_seed(99)).to(w) * w.std()
        m.head.bias[FG().to(w.device)] = 0


def sched_opt(ps, steps, lr=1e-5):
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    return opt, sch


def prelogits(m, x):
    return m.forward_head(m.forward_features(x), pre_logits=True)


def pgd(nat, nrm, x01, eps=0.03, step=0.01, its=3):
    """l_inf PGD (random start) against the original model on its own prediction; x01 in [0,1]"""
    with torch.no_grad():
        y0 = nat(nrm(x01)).argmax(-1)
    d = (torch.rand_like(x01) * 2 - 1) * eps
    for _ in range(its):
        d.requires_grad_(True)
        loss = F.cross_entropy(nat(nrm((x01 + d).clamp(0, 1))).float(), y0)
        g, = torch.autograd.grad(loss, d)
        d = (d.detach() + step * g.sign()).clamp(-eps, eps)
    return (x01 + d).clamp(0, 1).detach()


def train(m, nat, nrm, xd, xr, yr, kind, steps, seed=0):
    ps = list(m.parameters())
    for q in ps:
        q.requires_grad_(True)
    opt, sch = sched_opt(ps, steps)
    g = torch.Generator().manual_seed(seed)
    nondog = torch.tensor([c for c in range(1000) if c not in set(FG().tolist())])
    byc = [torch.nonzero(xd_labels == c)[:, 0] for c in FG().tolist() if (xd_labels == c).sum() >= 2]
    m.train()
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        if kind == "RL":
            i = torch.randint(len(xd), (S.BS // 2,), generator=g)
            j = torch.randint(len(xr), (S.BS // 2,), generator=g)
            xb = nrm(S.augment(torch.cat([xd[i], xr[j]]).to(S.dev), g))
            yb = torch.cat([nondog[torch.randint(len(nondog), (S.BS // 2,), generator=g)], yr[j]]).to(S.dev)
            for s in range(0, S.BS, S.MB):
                (F.cross_entropy(m(xb[s:s + S.MB]).float(), yb[s:s + S.MB], reduction="sum") / S.BS).backward()
        elif kind in ("DELETE", "Spotter"):
            if kind == "Spotter":  # pairs of same-class images so that every micro-batch has same-class pairs for L_sim
                cls = spotter_classes(len(byc), S.BS // 2, g)
                i = spotter_images(byc, cls, g)
            else:
                i = torch.randint(len(xd), (S.BS,), generator=g)
            x01 = S.augment(xd[i].to(S.dev), g)
            for s in range(0, S.BS, S.MB):
                xs = x01[s:s + S.MB]
                with torch.no_grad():
                    lt = masked_logp(nat(nrm(xs)))
                if kind == "DELETE":
                    ls = m(nrm(xs)).float().log_softmax(-1)
                    ((lt.exp() * (lt - ls)).sum(-1).sum() / S.BS).backward()
                    continue
                xp = pgd(nat, nrm, xs)
                with torch.no_grad():
                    lp = masked_logp(nat(nrm(xp)))
                h = m.forward_features(nrm(xs))
                f = m.forward_head(h, pre_logits=True)
                ls = m.head(f).float().log_softmax(-1)
                lsp = m(nrm(xp)).float().log_softmax(-1)
                lu = (lt.exp() * (lt - ls)).sum(-1).sum() / S.BS
                lo = (lp.exp() * (lp - lsp)).sum(-1).sum() / S.BS
                yb = torch.as_tensor(xd_labels[i[s:s + S.MB]]).to(S.dev)
                fn = F.normalize(f.float(), dim=-1)
                cs = fn @ fn.t()
                same = (yb[:, None] == yb[None, :]) & ~torch.eye(len(yb), dtype=torch.bool, device=S.dev)
                lsim = (cs[same].sum() / same.sum().clamp_min(1)) * len(yb) / S.BS
                (0.7 * lu + 0.3 * lo + 1.0 * lsim).backward()
        elif kind == "SCRUB":
            if t < min(1000, steps // 3 + 1):
                i = torch.randint(len(xd), (S.BS // 2,), generator=g)
                xb = nrm(S.augment(xd[i].to(S.dev), g))
                for s in range(0, len(xb), S.MB):
                    with torch.no_grad():
                        lt = nat(xb[s:s + S.MB]).float().log_softmax(-1)
                    ls = m(xb[s:s + S.MB]).float().log_softmax(-1)
                    (-(lt.exp() * (lt - ls)).sum(-1).sum() / S.BS).backward()
            j = torch.randint(len(xr), (S.BS,), generator=g)
            xb, yb = nrm(S.augment(xr[j].to(S.dev), g)), yr[j].to(S.dev)
            for s in range(0, S.BS, S.MB):
                with torch.no_grad():
                    lt = nat(xb[s:s + S.MB]).float().log_softmax(-1)
                z = m(xb[s:s + S.MB]).float()
                ls = z.log_softmax(-1)
                (((lt.exp() * (lt - ls)).sum(-1).sum() + F.cross_entropy(z, yb[s:s + S.MB], reduction="sum")) / S.BS).backward()
        opt.step()
        sch.step()
    return p18.frozen(m)


xd_labels = None


def repair(m, nat, nrm, xr, steps):
    class Masked(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            z = self.n(x).float().clone()
            z[:, FG().to(z.device)] = -1e4
            return z
    ps = wrap(m)
    p18.kd_train(m, Masked(nat), nrm, xr, ps, steps, 1e-3, seed=0)
    fold(m)
    return p18.frozen(m)


@torch.no_grad()
def feats(m, nrm, x, bs=100):
    return torch.cat([prelogits(m, nrm(x[s:s + bs].to(S.dev).float().div_(255))).float().cpu() for s in range(0, len(x), bs)])


def probe(m, nrm, xd, yd, xF, yF):
    cls = {int(c): i for i, c in enumerate(FG().tolist())}
    ftr, fte = feats(m, nrm, xd), feats(m, nrm, xF)
    mu, sd = ftr.mean(0), ftr.std(0).clamp_min(1e-6)
    ftr, fte = ((ftr - mu) / sd).to(S.dev), ((fte - mu) / sd).to(S.dev)
    ytr = torch.tensor([cls[int(c)] for c in yd]).to(S.dev)
    yte = torch.tensor([cls[int(c)] for c in yF]).to(S.dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(ftr.shape[1], len(cls)).to(S.dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(500):
            opt.zero_grad()
            F.cross_entropy(lin(ftr), ytr).backward()
            opt.step()
    return float((lin(fte).argmax(-1) == yte).float().mean())


@torch.no_grad()
def pra(m, nrm, xf, yf, xh, yh, xF, yF, xR, yR):
    five = torch.cat([torch.nonzero(yf == c)[:5, 0] for c in FG().tolist() if (yf == c).any()])
    f = feats(m, nrm, xf[five]).to(S.dev)
    cl = yf[five]
    hold_r = ~torch.isin(yh, FG())
    base = acc(m, nrm, xh[hold_r], yh[hold_r])
    W0, b0 = m.head.weight.detach().clone(), m.head.bias.detach().clone()
    rows = [int(c) for c in FG().tolist() if (cl == c).any()]
    P = torch.stack([F.normalize(f[cl == c].mean(0), dim=0) for c in rows]).to(W0)
    idx = torch.tensor(rows, device=W0.device)
    best = 0.0
    for a in [i / 10 for i in range(11)]:
        m.head.weight[idx] = a * P + (1 - a) * W0[idx]
        m.head.bias[idx] = (1 - a) * b0[idx]
        if base - acc(m, nrm, xh[hold_r], yh[hold_r]) <= 0.01:
            best = a
    m.head.weight[idx] = best * P + (1 - best) * W0[idx]
    m.head.bias[idx] = (1 - best) * b0[idx]
    out = dict(alpha=best, forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    m.head.weight.copy_(W0)
    m.head.bias.copy_(b0)
    return out


def main(group, model, arm):
    global xd_labels
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    xf, yf = S.cache("id_fit")
    xh, yh = S.cache("id_hold")
    x, y = S.cache("fresh")
    isf, hf, ff = torch.isin(y, FG()), torch.isin(yh, FG()), torch.isin(yf, FG())
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    xd, yd, xr, yr = xf[ff], yf[ff], xf[~ff], yf[~ff]
    xd_labels = yd
    res = dict(group=group, model=model, arm=arm, n_forget=int(isf.sum()), n_forget_fit=int(ff.sum()), recipe=dict(steps=STEPS, attack=ATT))

    def ev(mm):
        return dict(forget=acc(mm, nrm, xF, yF), retain=acc(mm, nrm, xR, yR),
                    forget_hold=acc(mm, nrm, xh[hf], yh[hf]), retain_hold=acc(mm, nrm, xh[~hf], yh[~hf]))
    res["native"] = ev(nat)
    if arm == "native":
        m = nat
    else:
        m = copy.deepcopy(nat)
        if arm.startswith("SD"):
            params = [p for b in range(12) for p in m.blocks[b].parameters()]
            If = fisher(m, nrm, params, xd[:400], yd[:400])
            Ir = fisher(m, nrm, params, xr[:800], yr[:800])
            res["dampened_fraction"] = dampen(params, If, Ir, float(arm[2:].split("-")[0]))
        if arm == "HRr" or arm.startswith("SD"):
            head_reinit(m)
            m = repair(p18.frozen(m), nat, nrm, xr, STEPS)
        elif arm in ("RL", "DELETE", "Spotter", "SCRUB"):
            m = train(m, nat, nrm, xd, xr, yr, arm, STEPS)
        else:
            raise ValueError(arm)
    res["final"] = ev(m)
    res["probe"] = probe(m, nrm, xd, yd, xF, yF)
    print(group, model, arm, "final", res["final"], "probe", res["probe"], flush=True)
    res["pra"] = pra(m, nrm, xf, yf, xh, yh, xF, yF, xR, yR)
    one = torch.tensor([int(torch.nonzero(yf == c)[0]) for c in FG().tolist() if (yf == c).any()])
    five = torch.cat([torch.nonzero(yf == c)[:5, 0] for c in FG().tolist() if (yf == c).any()])
    att = {}
    for name, xx, yy, st in (("few1", xf[one], yf[one], ATT // 3), ("few5", xf[five], yf[five], ATT), ("full_lora", xd, yd, ATT)):
        a_ = ce_attack(m, nrm, xx, yy, st)
        att[name] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR))
        del a_
        torch.cuda.empty_cache()
    att.update(strong_attacks(m, nrm, xf, yf, xF, yF, xR, yR))
    res["attacks"] = att
    print(group, model, arm, "pra", res["pra"], att, flush=True)
    json.dump(res, open(f"{OUT}/p108_{group}_{model}_{arm}.json", "w"), indent=1)
    print("done", group, model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
