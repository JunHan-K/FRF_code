"""PREREG79 (exploratory pilot): where should recovery happen? From the step-0 (damaged, untrained) network of a PREREG51
trajectory run (ViT-S, L/E/N, a/b), repair one cell k with the same KD recipe but only part of the network trainable:
  blk  : blocks.k only
  nb   : one other block of the same size (k+1, or k-1 for k = 11); parameter-matched control for blk
  rest : every parameter except blocks.k (secondary; more parameters)
Same data order, augmentation, lr schedule and number of steps (P79_STEPS, default 1000) for every strategy.
Outcome: KL to the original network on fresh (all images) after repair; also id_hold.
Step-0 predictors measured here (before any repair): squared-gradient norms of blocks.k, the nb block and the rest on
8 fixed calibration batches of id_fit; skip damage (KL on fresh[:1000] with blocks.k replaced by identity); swap gain
(KL on fresh[:1000] with blocks.k replaced by the native block k).
Step-0 networks: L/E = native blocks in the shuffled order (deit_perm.build); N = native weights plus the PREREG51 noise
(sigma from out/p51/<model>/D_N<p>_noise.json, generator seed 100 + index of p, as p47train.py).
KA (exit 1 on failure, no outputs): step-0 KL on id_hold equals the PREREG51 training record (ckpt step 0) to 2e-3.
usage: p79.py <worker> <n_workers>   (tasks = all model x kind x perm x cell x strategy, split by index)"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import s12r  # noqa: E402
from p60 import evaluate  # noqa: E402

SMOKE = int(os.environ.get("P79_SMOKE", 0))
OUT = "out/p79" + ("smoke" if SMOKE else "")
P51 = "out/p51"
MODELS, KINDS, PERMS = ("deit_s", "deit3_s", "augreg_s"), "LEN", "ab"
STRATS = ("blk", "nb", "rest")
STEPS = int(os.environ.get("P79_STEPS", 1000))
dev = S.dev


def step0(model, kind, p):
    D.M = s12r.MODELS[model]
    run = f"D_{kind}{p}"
    if kind == "N":
        D.KEEP[run] = list(range(12))
    m = D.build(run)
    if kind == "N":
        sigma = json.load(open(f"{P51}/{model}/{run}_noise.json"))["sigma"]
        teacher = S.load(D.M)
        clean = {n: t.detach().clone() for n, t in teacher.state_dict().items()
                 if any(n.startswith(f"blocks.{k}.") for k in range(6, 12))}
        gen = torch.Generator().manual_seed(100 + "abcdefghijklmnopqr".index(p))
        eps = {n: torch.randn(t.shape, generator=gen).to(t.device) * t.float().std() for n, t in clean.items()
               if t.is_floating_point() and t.numel() > 1}
        sd = m.state_dict()
        for n, e in eps.items():
            sd[n] = clean[n] + sigma * e.to(sd[n].dtype)
        m.load_state_dict(sd)
    return run, m.to(dev)


def cells(model, kind, p):
    run = f"D_{kind}{p}"
    keep = list(range(12)) if kind == "N" else D.KEEP[run]
    return [k for k in range(6, 12) if (keep[k] != k if kind == "L" else True)]


def tasks():
    out = []
    for model in MODELS:
        for kind in KINDS:
            for p in PERMS:
                D.M = s12r.MODELS[model]
                for k in cells(model, kind, p):
                    out += [(model, kind, p, k, "pre")] + [(model, kind, p, k, s) for s in STRATS]
    return out[:4] if SMOKE else out


def nb_of(k):
    return k + 1 if k < 11 else k - 1


def select(name, k, strat):
    if strat == "blk":
        return name.startswith(f"blocks.{k}.")
    if strat == "nb":
        return name.startswith(f"blocks.{nb_of(k)}.")
    return not name.startswith(f"blocks.{k}.")


def kl_to(m, teacher, nrm, x):
    m.eval()
    with torch.no_grad():
        kl = 0.0
        for s in range(0, len(x), 128):
            xb = nrm(x[s:s + 128].to(dev).float().div_(255))
            lt, lm = teacher(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
            kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
    return kl / len(x)


def pre(model, kind, p, k, teacher, nrm, data):
    run, m = step0(model, kind, p)
    xf, _ = S.cache("id_fit")
    g = torch.Generator().manual_seed(790)
    sq = {"blk": 0.0, "nb": 0.0, "rest": 0.0}
    m.eval()
    for _ in range(8):
        idx = torch.randint(len(xf), (64,), generator=g)
        x = nrm(xf[idx].to(dev).float().div_(255))
        m.zero_grad(set_to_none=True)
        lps = m(x).log_softmax(-1)
        with torch.no_grad():
            lpt = teacher(x).log_softmax(-1)
        ((lpt.exp() * (lpt - lps)).sum(-1).mean()).backward()
        for n, q in m.named_parameters():
            if q.grad is None:
                continue
            v = float(q.grad.pow(2).sum())
            sq["blk"] += v if select(n, k, "blk") else 0.0
            sq["nb"] += v if select(n, k, "nb") else 0.0
            sq["rest"] += v if select(n, k, "rest") else 0.0
    x1k = data["fresh"][0][:1000]
    r = dict(kl0_hold=evaluate(m, teacher, nrm, {"id_hold": data["id_hold"]})["id_hold"]["kl"],
             kl0_fresh1k=kl_to(m, teacher, nrm, x1k), sq=sq)
    blk = m.blocks[k]
    m.blocks[k] = torch.nn.Identity()
    r["kl_skip"] = kl_to(m, teacher, nrm, x1k)
    m.blocks[k] = teacher.blocks[k]
    r["kl_swap"] = kl_to(m, teacher, nrm, x1k)
    m.blocks[k] = blk
    r["n_params"] = {s: sum(q.numel() for n, q in m.named_parameters() if select(n, k, s)) for s in STRATS}
    r["ka_ref"] = json.load(open(f"{P51}/{model}/{run}.json"))["ckpt"][0]["kl_teacher"]
    return r


def repair(model, kind, p, k, strat, teacher, nrm, data):
    run, m = step0(model, kind, p)
    torch.manual_seed(0)
    named = list(m.named_parameters())
    for n, q in named:
        q.requires_grad_(select(n, k, strat))
    ps = [q for n, q in named if select(n, k, strat)]
    opt = torch.optim.AdamW(ps, lr=S.LR, weight_decay=S.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, STEPS) / STEPS)))
    xf, _ = S.cache("id_fit")
    g = torch.Generator().manual_seed(0)
    m.train()
    for t in range(1, STEPS + 1):
        idx = torch.randint(len(xf), (S.BS,), generator=g)
        x = nrm(S.augment(xf[idx].to(dev, non_blocking=True), g))
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            lps = m(x[s:s + S.MB]).log_softmax(-1)
            with torch.no_grad():
                lpt = teacher(x[s:s + S.MB]).log_softmax(-1)
            ((lpt.exp() * (lpt - lps)).sum(-1).sum() / S.BS).backward()
        opt.step()
        sched.step()
    return dict(eval=evaluate(m, teacher, nrm, data), n_params=sum(q.numel() for q in ps), steps=STEPS)


def main(w, nw):
    os.makedirs(OUT, exist_ok=True)
    data = {n: S.cache(n) for n in ("id_hold", "fresh")}
    cur, teacher, nrm = None, None, None
    for i, (model, kind, p, k, strat) in enumerate(tasks()):
        if i % nw != w:
            continue
        fn = f"{OUT}/p79_{model}_{kind}_{p}_{k}_{strat}.json"
        if os.path.exists(fn):
            continue
        if cur != model:
            D.M = s12r.MODELS[model]
            teacher = S.load(D.M).eval().to(dev)
            for q in teacher.parameters():
                q.requires_grad_(False)
            nrm = S.Norm(teacher).to(dev)
            cur = model
        D.M = s12r.MODELS[model]
        t0 = time.time()
        if strat == "pre":
            r = pre(model, kind, p, k, teacher, nrm, data)
            if abs(r["kl0_hold"] - r["ka_ref"]) > 2e-3:
                print("KA FAIL step0", model, kind, p, k, r["kl0_hold"], r["ka_ref"], flush=True)
                sys.exit(1)
        else:
            r = repair(model, kind, p, k, strat, teacher, nrm, data)
        r.update(model=model, kind=kind, perm=p, slot=k, strat=strat)
        json.dump(r, open(fn, "w"), indent=1)
        print(i, model, kind, p, k, strat, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
