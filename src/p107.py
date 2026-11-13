"""PREREG107 (GPU): seed replicates of HRr and OURS (PREREG104 arms) with repair seeds 1 and 2 (LoRA init and data order of the
dog-free repair; the selective-dampening step is deterministic) and the same seed for the attacks' data order.
Per network: final dog / retain on fresh; linear probe (as PREREG104); LoRA attacks few1 / few5 / full_lora (as PREREG104, attack
seed = s); the PREREG106 strong all-parameter attacks (seed = s). Seed 0 is PREREG104 + PREREG106.
Written 2026-11-08 08:37 KST, before any PREREG107 network exists.
usage: p107.py <model> <arm> <seed>"""
import copy
import json
import os
import sys

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import FORGET, Masked  # noqa: E402
from p99 import wrap, fold  # noqa: E402
from p100 import fisher, dampen  # noqa: E402
from p102 import head_reinit  # noqa: E402
from p104 import OURS, probe  # noqa: E402
from p106 import strong_attacks  # noqa: E402

OUT = os.environ.get("P107_OUT", "out/p107")
STEPS = int(os.environ.get("P107_STEPS", 3000))  # paper: default value (override only for quick tests)
ATT = int(os.environ.get("P107_ATT", 300))


def lora_attack(m, nrm, x, y, steps, seed):
    a = copy.deepcopy(m)
    torch.manual_seed(seed)
    ps = wrap(a)
    for q in a.parameters():
        q.requires_grad_(False)
    for q in ps:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    a.train()
    for _ in range(steps):
        idx = torch.randint(len(x), (S.BS,), generator=g)
        xb = nrm(S.augment(x[idx].to(S.dev, non_blocking=True), g))
        yb = y[idx].to(S.dev)
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            (nn.functional.cross_entropy(a(xb[s:s + S.MB]).float(), yb[s:s + S.MB], reduction="sum") / S.BS).backward()
        opt.step()
    return p18.frozen(a)


def main(model, arm, seed):
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    xf, yf = S.cache("id_fit")
    x, y = S.cache("fresh")
    isf, ff = torch.isin(y, FORGET), torch.isin(yf, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    xd, yd, xr, yr = xf[ff], yf[ff], xf[~ff], yf[~ff]
    res = dict(model=model, arm=arm, seed=seed, native=dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR)))
    m = copy.deepcopy(nat)
    a = OURS[model] if arm == "OURS" else arm
    res["built"] = a
    if a.startswith("SD"):
        params = [p for b in range(12) for p in m.blocks[b].parameters()]
        If = fisher(m, nrm, params, xd[:400], yd[:400])
        Ir = fisher(m, nrm, params, xr[:800], yr[:800])
        res["dampened_fraction"] = dampen(params, If, Ir, float(a[2:].split("-")[0]))
    elif a != "HRr":
        raise ValueError(arm)
    head_reinit(m)
    m = p18.frozen(m)
    torch.manual_seed(seed)
    ps = wrap(m)
    p18.kd_train(m, Masked(nat), nrm, xr, ps, STEPS, 1e-3, seed=seed)
    fold(m)
    m = p18.frozen(m)
    res["final"] = dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR))
    res["probe"] = probe(m, nrm, xd, yd, xF, yF)
    print(model, arm, seed, "final", res["final"], "probe", res["probe"], flush=True)
    one = torch.tensor([int(torch.nonzero(yf == c)[0]) for c in FORGET.tolist() if (yf == c).any()])
    five = torch.cat([torch.nonzero(yf == c)[:5, 0] for c in FORGET.tolist() if (yf == c).any()])
    att = {}
    for name, xx, yy, st in (("few1", xf[one], yf[one], ATT // 3), ("few5", xf[five], yf[five], ATT), ("full_lora", xd, yd, ATT)):
        a_ = lora_attack(m, nrm, xx, yy, st, seed)
        att[name] = dict(forget=acc(a_, nrm, xF, yF), retain=acc(a_, nrm, xR, yR))
        del a_
    att.update(strong_attacks(m, nrm, xf, yf, xF, yF, xR, yR, seed=seed))
    res["attacks"] = att
    print(model, arm, seed, att, flush=True)
    json.dump(res, open(f"{OUT}/p107_{model}_{arm}_s{seed}.json", "w"), indent=1)
    print("done", model, arm, seed, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
