"""PREREG115 (GPU, evaluation only): rewinding single blocks of the unlearned network to native. Which blocks hold the forgetting?
Networks (dogs, deit_s / deit3_s / augreg_s): OURS (primary), HRr, RL (saved PREREG104 weights), SCRUB (retrained, PREREG108, seed 0).
Readout head for every variant = the network's own head with the forget-group rows (weight and bias) copied from the native head
(OURS / HRr re-initialise those rows; without this no block rewind could bring dogs back).
Variants: base (no block rewound); single(k): block k replaced by the native block k, k = 0..11; deep(j): blocks j..11 rewound,
j = 11..0 (j = 0 = all blocks native). Measured: fresh dog and retained-class top-1 of every variant.
Written 2026-11-11, before any PREREG115 value exists.
usage: p115.py <model> <arm>"""
import copy
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
from p110 import load_arm  # noqa: E402

OUT = os.environ.get("P115_OUT", "out/p115")


@torch.no_grad()
def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M).to(S.dev))
    nrm = S.Norm(nat).to(S.dev)
    with torch.enable_grad():
        U = load_arm(model, arm, nat, nrm)
    U = copy.deepcopy(U)
    f = FORGET.to(S.dev)
    U.head.weight[f] = nat.head.weight[f].to(U.head.weight)
    U.head.bias[f] = nat.head.bias[f].to(U.head.bias)
    x, y = S.cache("fresh")
    isf = torch.isin(y, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    own = list(U.blocks)

    def ev(rew):
        for k in range(12):
            U.blocks[k] = nat.blocks[k] if k in rew else own[k]
        r = dict(forget=acc(U, nrm, xF, yF), retain=acc(U, nrm, xR, yR))
        for k in range(12):
            U.blocks[k] = own[k]
        return r
    res = dict(model=model, arm=arm, native=dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR)),
               base=ev(set()), single={}, deep={})
    print(model, arm, "native", res["native"], "base", res["base"], flush=True)
    for k in range(12):
        res["single"][k] = ev({k})
        print(model, arm, "single", k, res["single"][k], flush=True)
    for j in range(11, -1, -1):
        res["deep"][j] = ev(set(range(j, 12)))
        print(model, arm, "deep", j, res["deep"][j], flush=True)
    res["complete"] = True
    json.dump(res, open(f"{OUT}/p115_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
