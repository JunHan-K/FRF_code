"""PREREG112 (GPU): relearning attack that also keeps the retained classes (mixed attack), dogs, ViT-S deit_s / deit3_s / augreg_s.
Networks: HRr, OURS, RL (saved PREREG104 fp16 weights) and SCRUB (retrained, PREREG108 recipe, seed 0); native numbers are read from
the frozen native network in the same run.
Attack (each from a fresh copy; all parameters incl. head; AdamW, no decay, warmup + cosine, augmentation, hard labels, seed 0):
every batch of 64 = 32 random dog id_fit images + 32 random other-class id_fit images. Settings: lr 1e-4 and lr 3e-4, 1,000 steps.
Measured: fresh dog and retained-class top-1 after each attack.
Written 2026-11-10, before any PREREG112 value exists.
usage: p112.py <model> <arm>"""
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
from p110 import load_arm  # noqa: E402

OUT = os.environ.get("P112_OUT", "out/p112")
ATTACKS = [(1e-4, 1000), (3e-4, 1000)]


def mixed_ft(m, nrm, xd, yd, xr, yr, steps, lr, seed=0):
    a = copy.deepcopy(m)
    ps = list(a.parameters())
    for q in ps:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    warm = min(S.WARM, max(steps // 10, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(seed)
    a.train()
    h = S.BS // 2
    for _ in range(steps):
        i = torch.randint(len(xd), (h,), generator=g)
        j = torch.randint(len(xr), (h,), generator=g)
        xb = nrm(S.augment(torch.cat([xd[i], xr[j]]).to(S.dev, non_blocking=True), g))
        yb = torch.cat([yd[i], yr[j]]).to(S.dev)
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            (F.cross_entropy(a(xb[s:s + S.MB]).float(), yb[s:s + S.MB], reduction="sum") / S.BS).backward()
        opt.step()
        sched.step()
    return p18.frozen(a)


def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M).to(S.dev))
    nrm = S.Norm(nat).to(S.dev)
    m = load_arm(model, arm, nat, nrm)
    xf, yf = S.cache("id_fit")
    x, y = S.cache("fresh")
    isf = torch.isin(y, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    ff = torch.isin(yf, FORGET)
    res = dict(model=model, arm=arm, native=dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR)),
               start=dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR)), attacks={})
    print(model, arm, "start", res["start"], flush=True)
    for lr, st in ATTACKS:
        a = mixed_ft(m, nrm, xf[ff], yf[ff], xf[~ff], yf[~ff], st, lr)
        r = dict(lr=lr, steps=st, forget=acc(a, nrm, xF, yF), retain=acc(a, nrm, xR, yR))
        res["attacks"][f"lr{lr:g}_s{st}"] = r
        print(model, arm, r, flush=True)
        del a
        torch.cuda.empty_cache()
        json.dump(res, open(f"{OUT}/p112_{model}_{arm}.json", "w"), indent=1)
    res["complete"] = True
    json.dump(res, open(f"{OUT}/p112_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
