"""PREREG106 (GPU): a strong all-parameter relearning attack on the 21 PREREG104 networks (saved fp16 weights, out/p104w; native
loaded from timm). Attacks (each from a copy, ALL parameters incl. head, AdamW, no decay, batch 64, warmup + cosine, same augmentation,
hard labels on dog id_fit images, seed 0):
  sft1   lr 1e-4, one image per dog class, 100 steps
  sft5   lr 1e-4, five images per dog class, 300 steps
  sftall lr 1e-4, all 938 dog id_fit images, 1,000 steps
Measured: dog and retain top-1 on the fresh set after each attack.
Written 2026-11-08 08:36 KST, before any PREREG106 value exists.
usage: p106.py <model> <arm>"""
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

OUT = os.environ.get("P106_OUT", "out/p106")
W = "out/p104w"
SCALE = float(os.environ.get("P106_SCALE", 1.0))  # smoke test only (multiplies step counts)


def strong_ft(m, nrm, x, y, steps, lr=1e-4, seed=0):
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
    for _ in range(steps):
        i = torch.randint(len(x), (S.BS,), generator=g)
        xb = nrm(S.augment(x[i].to(S.dev, non_blocking=True), g))
        yb = y[i].to(S.dev)
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            (F.cross_entropy(a(xb[s:s + S.MB]).float(), yb[s:s + S.MB], reduction="sum") / S.BS).backward()
        opt.step()
        sched.step()
    return p18.frozen(a)


def strong_attacks(m, nrm, xf, yf, xF, yF, xR, yR, seed=0):
    ff = torch.isin(yf, FORGET)
    one = torch.tensor([int(torch.nonzero(yf == c)[0]) for c in FORGET.tolist() if (yf == c).any()])
    five = torch.cat([torch.nonzero(yf == c)[:5, 0] for c in FORGET.tolist() if (yf == c).any()])
    out = {}
    for name, idx, steps in (("sft1", one, 100), ("sft5", five, 300), ("sftall", torch.nonzero(ff)[:, 0], 1000)):
        a = strong_ft(m, nrm, xf[idx], yf[idx], max(int(steps * SCALE), 1), seed=seed)
        out[name] = dict(forget=acc(a, nrm, xF, yF), retain=acc(a, nrm, xR, yR))
        del a
        torch.cuda.empty_cache()
    return out


def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    m = S.load(D.M)
    nat_sd = None
    if arm != "native":
        sd = torch.load(f"{W}/{model}_{arm}.pt", map_location="cpu")
        nat_sd = m.state_dict()
        m.load_state_dict({k: v.to(nat_sd[k].dtype) for k, v in sd.items()})
    m = p18.frozen(m.to(S.dev))
    nrm = S.Norm(m).to(S.dev)
    xf, yf = S.cache("id_fit")
    x, y = S.cache("fresh")
    isf = torch.isin(y, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    res = dict(model=model, arm=arm, scale=SCALE, start=dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR)))
    print(model, arm, "start", res["start"], flush=True)
    res["attacks"] = strong_attacks(m, nrm, xf, yf, xF, yF, xR, yR)
    print(model, arm, res["attacks"], flush=True)
    json.dump(res, open(f"{OUT}/p106_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
