"""PREREG60: removal and recovery outcomes on the same blocks (experiment E).
  rm  <model> <L|E|N> <perm> <slot> : parent = seed-0 recovered network. Remove the block in <slot>; evaluate immediately;
        then the same recovery budget as PREREG18 u1 / PREREG55 single (all parameters, 1000 KD steps to the original
        network, S.LR / S.WD, seed 0); evaluate again.
  ctl <model> <L|E|N> <perm>        : the parent, no removal, the same 1000 extra KD steps (no-removal control).
  resel <model> <crit>              : PREREG30 selections (k = 1..4) re-recovered with the PREREG32 cheap recipe (rank-1
        LoRA, 1000 steps, feature KD on, lr 1e-3, seed 0); top-1 (ImageNet-R / -A inside their official 200 classes), KL and agreement to the original.
Evaluation: id_hold (comparison with earlier work) and, with P60_FRESH=1, the fresh set (final evaluation). For each net:
top-1, KL(original || net), agreement with the original; rm / ctl also agreement with the parent.
usage: p60.py rm <model> <kind> <perm> <slot> | p60.py ctl <model> <kind> <perm> | p60.py resel <model> <crit>"""
import copy
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p11  # noqa: E402
import s12r  # noqa: E402
import p18  # noqa: E402
from p47 import e_paths  # noqa: E402

OUT = os.environ.get("P60_OUT", "out/p60")
STEPS = int(os.environ.get("P60_STEPS", 1000))  # PREREG63 D: alternative re-recovery budget
dev = S.dev


def sets():
    names = ["id_hold"] + (["fresh"] if os.environ.get("P60_FRESH") else [])
    return {n: S.cache(n) for n in names}


@torch.no_grad()
def evaluate(m, nat, nrm, data, parent=None):
    m.eval()
    out = {}
    for n, (x, y) in data.items():
        kl, ag, ok, agp = 0.0, 0, 0, 0
        for s in range(0, len(x), 128):
            xb = nrm(x[s:s + 128].to(dev).float().div_(255))
            lt, lm = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
            kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
            ag += int((lt.argmax(-1) == lm.argmax(-1)).sum())
            ok += int((lm.argmax(-1).cpu() == y[s:s + 128]).sum())
            if parent is not None:
                agp += int((parent(xb).argmax(-1) == lm.argmax(-1)).sum())
        out[n] = dict(top1=ok / len(x), kl=kl / len(x), agree=ag / len(x))
        if parent is not None:
            out[n]["agree_parent"] = agp / len(x)
    return out


def parent_net(model, kind, p):
    if kind == "L":
        run, pt, _ = p11.load_case(model, p)
    else:
        D.M = s12r.MODELS[model]
        run, pt, _ = e_paths(model, p, kind)
    m = D.build(run)
    m.load_state_dict(torch.load(pt, map_location=dev))
    return run, p18.frozen(m)


def rm(model, kind, p, slot=None):
    D.M = s12r.MODELS[model]
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    data = sets()
    xf, _ = S.cache("id_fit")
    run, par = parent_net(model, kind, p)
    r = dict(model=model, kind=kind, perm=p, run=run, slot=None if slot is None else int(slot), steps=STEPS,
             parent=evaluate(par, nat, nrm, data))
    m = copy.deepcopy(par)
    if slot is not None:
        k = int(slot)
        r["src"] = D.KEEP[run][k]
        m.blocks = nn.Sequential(*[b for t, b in enumerate(m.blocks) if t != k])
        r["immediate"] = evaluate(p18.frozen(m), nat, nrm, data, par)
    for q in m.parameters():
        q.requires_grad_(True)
    p18.kd_train(m, nat, nrm, xf, list(m.parameters()), STEPS, S.LR, S.WD)
    r["recovered"] = evaluate(p18.frozen(m), nat, nrm, data, par)
    os.makedirs(OUT, exist_ok=True)
    tag = f"rm_{model}_{kind}_{p}_{slot}" if slot is not None else f"ctl_{model}_{kind}_{p}"
    json.dump(r, open(f"{OUT}/{tag}.json", "w"), indent=1)
    print(json.dumps(r), flush=True)


def resel(model, crit):
    D.M = s12r.MODELS[model]
    src = json.load(open(f"{p18.OUT}/p23_{model}_{crit}_lora1_fix_main.json"))
    cheap = json.load(open(f"{p18.OUT}/p23_{model}_{crit}_lora1_fix_cheap.json"))
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    data = sets()
    ev = p18.eval_sets()
    xf, _ = S.cache("id_fit")
    r = dict(model=model, crit=crit, steps=[])
    for st in src["steps"]:
        m = copy.deepcopy(nat)
        m.blocks = nn.Sequential(*[b for t, b in enumerate(m.blocks) if t not in st["drop"]])
        ps = []
        for blk in m.blocks:
            ps += p18.lora_wrap(blk, 1)
        p18.kd_train(m, nat, nrm, xf, ps, 1000, 1e-3, seed=0, featkd=True)
        m = p18.frozen(m)
        stored = next((c["acc"] for c in cheap["steps"] if c["k"] == st["k"]), None)  # cheap recipe has k = 2..4 only
        r["steps"].append(dict(k=st["k"], drop=st["drop"], acc={n: p18.top1(m, nrm, *v) for n, v in ev.items()},
                               stored_cheap=stored, metrics=evaluate(m, nat, nrm, data)))
        del m
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(f"{OUT}/resel_{model}_{crit}.json", "w"), indent=1)
    print(json.dumps(r)[:1500], flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "rm":
        rm(*a[1:])
    elif a[0] == "ctl":
        rm(*a[1:4])
    else:
        resel(*a[1:])
