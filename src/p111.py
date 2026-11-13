"""PREREG111 (GPU): relearning under attacks of different budgets (dogs, ViT-S deit_s / deit3_s / augreg_s).
Networks: native, HRr, OURS, RL from the saved PREREG104 fp16 weights (out/p104w); SCRUB retrained here (PREREG108 recipe, seed 0).
Attacks (each from a fresh copy; all parameters incl. head; all 938 dog id_fit images; AdamW, no decay, batch 64, warmup + cosine,
augmentation, hard labels, seed 0 = p106.strong_ft):
  lr 1e-4 with 100, 300, 1,000, 3,000 steps;  lr 3e-5 and 3e-4 with 1,000 steps.
Measured: dog and retained-class top-1 on the fresh set after each attack.
Written 2026-11-09, before any PREREG111 value exists.
usage: p111.py <model> <arm>"""
import json
import os
import sys

import torch  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import FORGET  # noqa: E402
from p106 import strong_ft  # noqa: E402
from p110 import load_arm  # noqa: E402

OUT = os.environ.get("P111_OUT", "out/p111")
SCALE = float(os.environ.get("P111_SCALE", 1.0))  # paper: default value (override only for quick tests)
ATTACKS = [(1e-4, 100), (1e-4, 300), (1e-4, 1000), (1e-4, 3000), (3e-5, 1000), (3e-4, 1000)]


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
    ff = torch.nonzero(torch.isin(yf, FORGET))[:, 0]
    res = dict(model=model, arm=arm, scale=SCALE, n_attack_images=len(ff),
               native=dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR)),
               start=dict(forget=acc(m, nrm, xF, yF), retain=acc(m, nrm, xR, yR)), attacks={})
    print(model, arm, "start", res["start"], flush=True)
    for lr, st in ATTACKS:
        a = strong_ft(m, nrm, xf[ff], yf[ff], max(int(st * SCALE), 1), lr=lr)
        r = dict(lr=lr, steps=st, forget=acc(a, nrm, xF, yF), retain=acc(a, nrm, xR, yR))
        res["attacks"][f"lr{lr:g}_s{st}"] = r
        print(model, arm, r, flush=True)
        del a
        torch.cuda.empty_cache()
        json.dump(res, open(f"{OUT}/p111_{model}_{arm}.json", "w"), indent=1)
    res["complete"] = True
    json.dump(res, open(f"{OUT}/p111_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
