"""PREREG114 (GPU): FRF after the relearning attack. Does the attack restore the blocks (SCRUB) or not (OURS)?
Networks (dogs, ViT-S deit_s / deit3_s / augreg_s): native (control: the attack's own departure), HRr, OURS, RL (saved PREREG104
weights), SCRUB (retrained, PREREG108 recipe, seed 0). Attack = sftall (p106.strong_ft, lr 1e-4, 1,000 steps, all parameters,
all 938 dog id_fit images, seed 0). FRF exactly as PREREG110 (p110.frf: block k of the network into slot k of the frozen native
network, references = twelve native blocks, Fisher metric of the native prediction, Gram pooled over images) on the same F / R sets,
before and after the attack, in the same run.
Written 2026-11-11, before any PREREG114 value exists.
usage: p114.py <model> <arm>"""
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
from p110 import load_arm, frf  # noqa: E402

OUT = os.environ.get("P114_OUT", "out/p114")
STEPS = int(os.environ.get("P114_STEPS", 1000))  # paper: default value (override only for quick tests)


def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M).to(S.dev))
    nrm = S.Norm(nat).to(S.dev)
    U = load_arm(model, arm, nat, nrm)
    xf, yf = S.cache("id_fit")
    x, y = S.cache("fresh")
    isf = torch.isin(y, FORGET)
    xF, yF, xR_all, yR_all = x[isf], y[isf], x[~isf], y[~isf]
    ri = torch.randperm(len(xR_all), generator=torch.Generator().manual_seed(0))[:len(xF)]
    ff = torch.nonzero(torch.isin(yf, FORGET))[:, 0]
    res = dict(model=model, arm=arm, steps=STEPS,
               native=dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR_all, yR_all)))
    for tag, net in (("pre", U), ("post", None)):
        if net is None:
            net = strong_ft(U, nrm, xf[ff], yf[ff], STEPS, lr=1e-4)
        r = dict(forget=acc(net, nrm, xF, yF), retain=acc(net, nrm, xR_all, yR_all))
        r["F"] = frf(nat, net, nrm, xF)
        r["R"] = frf(nat, net, nrm, xR_all[ri])
        res[tag] = r
        print(model, arm, tag, r["forget"], r["retain"], "F nu", [round(q["nu"], 3) for q in r["F"]],
              "R nu", [round(q["nu"], 3) for q in r["R"]], flush=True)
        json.dump(res, open(f"{OUT}/p114_{model}_{arm}.json", "w"), indent=1)
    res["complete"] = True
    json.dump(res, open(f"{OUT}/p114_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
