"""PREREG104 (GPU): baselines from the unlearning literature and stronger relearning attacks, ViT-S (deit_s, deit3_s, augreg_s).
Forget set, data, evaluation as PREREG102. Arms:
  HRr       head re-init + dog-free repair (PREREG102 reference, rebuilt with the same code)
  OURS      the arm PREREG102 selected per model (deit_s SD3-12H, deit3_s SD10-12H, augreg_s SD10-12H), rebuilt
  FT        fine-tune all parameters on retain id_fit images, hard labels (no dog image)
  RL        random labels: all parameters, batches of 32 dog images with random non-dog labels + 32 retain images
  SalUn     RL restricted to the 50% of parameters with the largest dog-loss gradient saliency (Fan et al. 2024)
  SSD       selective synaptic dampening on all parameters, alpha 10, lambda 1, no repair (Foster et al. 2024)
FT / RL / SalUn: AdamW lr 1e-5, no decay, batch 64, 3,000 steps, warmup + cosine, same augmentation.
Attacks on every final network (each from a copy): few1 (rank-4 LoRA on blocks, head frozen, 1 dog image / class, 100 steps; as
PREREG99-103), few5 (LoRA, 5 / class, 300 steps), full_lora (LoRA, all 938 dog id_fit images, 300 steps), full_ft (ALL parameters
incl. head, all dog images, AdamW 1e-5, 300 steps), few5_ft (all parameters, 5 / class, 300 steps); teacher (unmasked KD, LoRA).
Linear probe: 118-way dog classifier on frozen pre-logit features, fit on dog id_fit, top-1 on fresh dogs (native included).
Weights saved in fp16 to out/p104w for a later membership-inference test.
Written 2026-11-08 05:59 KST, before any PREREG104 network exists.
usage: p104.py <model> <arm>"""
import copy
import json
import math
import os
import sys

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import FORGET  # noqa: E402
from p99 import ce_attack, wrap, fold  # noqa: E402
from p100 import fisher, dampen  # noqa: E402
from p101 import repair  # noqa: E402
from p102 import head_reinit  # noqa: E402

OUT = os.environ.get("P104_OUT", "out/p104")
WOUT = os.environ.get("P104_WOUT", "out/p104w")
STEPS = int(os.environ.get("P104_STEPS", 3000))  # P104_STEPS / P104_ATT: smoke test only
ATT = int(os.environ.get("P104_ATT", 300))
OURS = {"deit_s": "SD3-12H", "deit3_s": "SD10-12H", "augreg_s": "SD10-12H"}
NONDOG = torch.tensor([c for c in range(1000) if c not in set(FORGET.tolist())])


def train_full(m, nrm, xr, yr, steps, xd=None, mask=None, lr=1e-5, seed=0):
    """all-parameter CE training; xd given -> half of each batch is dog images with random non-dog labels (RL / SalUn);
    mask: {param: bool tensor} -> only masked entries are updated"""
    ps = [q for q in m.parameters()]
    for q in ps:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(seed)
    m.train()
    nd = S.BS // 2 if xd is not None else 0
    for _ in range(steps):
        xs, ys = [], []
        if nd:
            i = torch.randint(len(xd), (nd,), generator=g)
            xs.append(xd[i]); ys.append(NONDOG[torch.randint(len(NONDOG), (nd,), generator=g)])
        j = torch.randint(len(xr), (S.BS - nd,), generator=g)
        xs.append(xr[j]); ys.append(yr[j])
        xb = nrm(S.augment(torch.cat(xs).to(S.dev, non_blocking=True), g))
        yb = torch.cat(ys).to(S.dev)
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            (F.cross_entropy(m(xb[s:s + S.MB]).float(), yb[s:s + S.MB], reduction="sum") / S.BS).backward()
        if mask is not None:
            for q in ps:
                if q.grad is not None:
                    q.grad.mul_(mask[q])
        opt.step()
        sched.step()
    return p18.frozen(m)


def saliency_mask(m, nrm, xd, yd, frac=0.5, n=400):
    ps = [q for q in m.parameters()]
    for q in ps:
        q.requires_grad_(True)
    m.eval()
    acc_g = [torch.zeros_like(q) for q in ps]
    for s in range(0, n, S.MB):
        z = m(nrm(xd[s:s + S.MB].to(S.dev).float().div_(255))).float()
        gs = torch.autograd.grad(F.cross_entropy(z, yd[s:s + S.MB].to(S.dev), reduction="sum"), ps)
        for a, g_ in zip(acc_g, gs):
            a += g_.detach()
    flat = torch.cat([a.abs().flatten() for a in acc_g])
    k = int(frac * flat.numel())
    thr = flat.kthvalue(flat.numel() - k).values
    return {q: (a.abs() > thr).float() for q, a in zip(ps, acc_g)}


def attack_ft(m, nrm, x, y, steps):
    a = copy.deepcopy(m)
    return train_full(a, nrm, x, y, steps, lr=1e-5, seed=0)


@torch.no_grad()
def feats(m, nrm, x, bs=100):
    out = []
    for s in range(0, len(x), bs):
        h = m.forward_features(nrm(x[s:s + bs].to(S.dev).float().div_(255)))
        out.append(m.forward_head(h, pre_logits=True).float().cpu())
    return torch.cat(out)


def probe(m, nrm, xd, yd, xF, yF):
    cls = {int(c): i for i, c in enumerate(FORGET.tolist())}
    ftr, fte = feats(m, nrm, xd), feats(m, nrm, xF)
    mu, sd = ftr.mean(0), ftr.std(0).clamp_min(1e-6)
    ftr, fte = ((ftr - mu) / sd).to(S.dev), ((fte - mu) / sd).to(S.dev)
    ytr = torch.tensor([cls[int(c)] for c in yd]).to(S.dev)
    yte = torch.tensor([cls[int(c)] for c in yF]).to(S.dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(ftr.shape[1], len(cls)).to(S.dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(500):
        opt.zero_grad()
        F.cross_entropy(lin(ftr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(fte).argmax(-1) == yte).float().mean())


def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WOUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    xf, yf = S.cache("id_fit")
    x, y = S.cache("fresh")
    isf, ff = torch.isin(y, FORGET), torch.isin(yf, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    xd, yd, xr, yr = xf[ff], yf[ff], xf[~ff], yf[~ff]
    res = dict(model=model, arm=arm, recipe=dict(steps=STEPS, attack=ATT))
    res["native"] = dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR))
    if arm == "native":
        m = nat
    else:
        m = copy.deepcopy(nat)
        a = OURS[model] if arm == "OURS" else arm
        res["built"] = a
        if a in ("HRr",) or a.startswith("SD") and a.endswith("H"):
            if a.startswith("SD"):
                params = [p for b in range(12) for p in m.blocks[b].parameters()]
                If = fisher(m, nrm, params, xd[:400], yd[:400])
                Ir = fisher(m, nrm, params, xr[:800], yr[:800])
                res["dampened_fraction"] = dampen(params, If, Ir, float(a[2:].split("-")[0]))
            head_reinit(m)
            m = p18.frozen(m)
            ps = wrap(m)
            repair(m, nat, nrm, xr, ps, STEPS, [])
            fold(m)
        elif a == "FT":
            m = train_full(m, nrm, xr, yr, STEPS)
        elif a == "RL":
            m = train_full(m, nrm, xr, yr, STEPS, xd=xd)
        elif a == "SalUn":
            mask = saliency_mask(m, nrm, xd, yd)
            res["salun_frac"] = float(sum(v.sum() for v in mask.values()) / sum(v.numel() for v in mask.values()))
            m = train_full(m, nrm, xr, yr, STEPS, xd=xd, mask=mask)
        elif a == "SSD":
            params = [p for p in m.parameters()]
            If = fisher(m, nrm, params, xd[:400], yd[:400])
            Ir = fisher(m, nrm, params, xr[:800], yr[:800])
            res["dampened_fraction"] = dampen(params, If, Ir, 10.0)
        else:
            raise ValueError(arm)
        m = p18.frozen(m)
        torch.save({k: v.half().cpu() for k, v in m.state_dict().items()}, f"{WOUT}/{model}_{arm}.pt")
    res["final"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    print(model, arm, "final", res["final"], flush=True)
    res["probe"] = probe(m, nrm, xd, yd, xF, yF)
    print(model, arm, "probe", res["probe"], flush=True)
    one = torch.tensor([int(torch.nonzero(yf == c)[0]) for c in FORGET.tolist() if (yf == c).any()])
    five = torch.cat([torch.nonzero(yf == c)[:5, 0] for c in FORGET.tolist() if (yf == c).any()])
    att = {}
    for name, fn in (("few1", lambda: ce_attack(m, nrm, xf[one], yf[one], ATT // 3)),
                     ("few5", lambda: ce_attack(m, nrm, xf[five], yf[five], ATT)),
                     ("full_lora", lambda: ce_attack(m, nrm, xd, yd, ATT)),
                     ("full_ft", lambda: attack_ft(m, nrm, xd, yd, ATT)),
                     ("few5_ft", lambda: attack_ft(m, nrm, xf[five], yf[five], ATT))):
        a_ = fn()
        att[name] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR))
        del a_
        torch.cuda.empty_cache()
    a_ = copy.deepcopy(m)
    p18.kd_train(a_, nat, nrm, xr, wrap(a_), ATT, 1e-3, seed=0)
    a_ = p18.frozen(a_)
    att["teacher"] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR))
    del a_
    res["attacks"] = att
    res["n_five"] = len(five)
    print(model, arm, att, flush=True)
    json.dump(res, open(f"{OUT}/p104_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
