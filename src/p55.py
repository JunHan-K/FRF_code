"""PREREG55: distinctness vs importance vs recoverability, and immediate vs recovered removal damage.
  single <model> <block> : drop one native block (blocks 1..10). Immediate: top-1, KL to the original and top-1 agreement
                           with the original on id_hold. Then the PREREG18 u1 recovery (all parameters, 1000 KD steps,
                           S.LR / S.WD, seed 0) and the same three metrics. Known answer: immediate top-1 equals u1's
                           top1_dropped_no_recovery; recovered top-1 is compared with u1's truth_top1.
  select <model> <crit>  : the PREREG30 selections (p23_<model>_<crit>_lora1_fix_main.json), k = 1..4, evaluated with no
                           recovery on id_hold / shift_R / shift_A (ImageNet-R / -A inside their official 200 classes) + KL and agreement on id_hold.
usage: p55.py single <model> <block> | p55.py select <model> <crit>"""
import copy
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import s12r  # noqa: E402
import p18  # noqa: E402

OUT = "out/p55"
dev = S.dev


@torch.no_grad()
def vs_native(m, nat, nrm, x):
    """KL(original || m) and top-1 agreement with the original"""
    kl, ag = 0.0, 0
    for s in range(0, len(x), 128):
        xb = nrm(x[s:s + 128].to(dev).float().div_(255))
        lt, ls = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
        kl += float((lt.exp() * (lt - ls)).sum(-1).sum())
        ag += int((lt.argmax(-1) == ls.argmax(-1)).sum())
    return kl / len(x), ag / len(x)


def drop(nat, blocks):
    m = copy.deepcopy(nat)
    m.blocks = nn.Sequential(*[b for t, b in enumerate(m.blocks) if t not in blocks])
    return m


def single(model, b):
    b = int(b)
    D.M = s12r.MODELS[model]
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    xf, _ = S.cache("id_fit")
    xh, yh = S.cache("id_hold")
    r = dict(model=model, block=b, native_top1=p18.top1(nat, nrm, xh, yh))
    m = p18.frozen(drop(nat, [b]))
    kl, ag = vs_native(m, nat, nrm, xh)
    r["immediate"] = dict(top1=p18.top1(m, nrm, xh, yh), kl=kl, agree=ag)
    m = drop(nat, [b])
    p18.kd_train(m, nat, nrm, xf, list(m.parameters()), 1000, S.LR, S.WD)
    m = p18.frozen(m)
    kl, ag = vs_native(m, nat, nrm, xh)
    r["recovered"] = dict(top1=p18.top1(m, nrm, xh, yh), kl=kl, agree=ag)
    u1 = f"{p18.OUT}/u1_{model}_{b}.json"
    if os.path.exists(u1):
        u = json.load(open(u1))
        r["u1"] = dict(top1_dropped_no_recovery=u["top1_dropped_no_recovery"], truth_top1=u["truth_top1"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(f"{OUT}/single_{model}_{b}.json", "w"), indent=1)
    print(json.dumps(r), flush=True)


def select(model, crit):
    D.M = s12r.MODELS[model]
    src = json.load(open(f"{p18.OUT}/p23_{model}_{crit}_lora1_fix_main.json"))
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    sets = p18.eval_sets()
    xh = sets["id_hold"][0]
    r = dict(model=model, crit=crit, order=src["order"], native={n: p18.top1(nat, nrm, *v) for n, v in sets.items()},
             native_stored=src["native"], steps=[])
    for st in src["steps"]:
        m = p18.frozen(drop(nat, st["drop"]))
        kl, ag = vs_native(m, nat, nrm, xh)
        r["steps"].append(dict(k=st["k"], drop=st["drop"], immediate={n: p18.top1(m, nrm, *v) for n, v in sets.items()},
                               kl=kl, agree=ag, recovered_main=st["acc"]))
        del m
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(f"{OUT}/select_{model}_{crit}.json", "w"), indent=1)
    print(json.dumps(r)[:1500], flush=True)


if __name__ == "__main__":
    {"single": single, "select": select}[sys.argv[1]](*sys.argv[2:])
