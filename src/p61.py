"""PREREG61 F: rewinding the training change of one block (2 x 2), on the PREREG51 trajectory networks (ViT-S, L/E/N, a/b).
theta_0 = step-0 checkpoint (for L/E the shuffled native blocks; for N the noised blocks), theta_T = final (step 3000).
For each cell k (L: relocated slots; E/N: slots 6..11):
  S11 = theta_T everywhere; S00 = theta_0 everywhere; S01 = theta_T with blocks.k from theta_0 (block untrained, rest
  trained); S10 = theta_0 with blocks.k from theta_T (block trained, rest untrained).
For a relocated block, "untrained" is the source block as it arrived in slot k, not the slot's native block.
Metrics: KL(original || net), top-1, agreement with the original; id_hold and, with P61_FRESH=1, fresh.
usage: p61.py <model> <L|E|N> <perm>"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p47 import e_paths  # noqa: E402
from p60 import evaluate  # noqa: E402

OUT = "out/p61"
dev = S.dev


@torch.no_grad()
def main(model, kind, p):
    D.M = s12r.MODELS[model]
    run = f"D_{kind}{p}"
    if kind == "N":
        e_paths(model, p, "N")
    d = f"out/p51/{model}"
    sT = torch.load(f"{d}/{run}.pt", map_location=dev)
    s0 = torch.load(f"{d}/{run}_ck0.pt", map_location=dev)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    data = {n: S.cache(n) for n in ["id_hold"] + (["fresh"] if os.environ.get("P61_FRESH") else [])}
    net = D.build(run).to(dev)

    def ev(sd):
        net.load_state_dict(sd)
        return evaluate(p18.frozen(net), nat, nrm, data)

    ks = [k for k in range(6, 12) if (D.KEEP[run][k] != k if kind == "L" else True)]
    r = dict(model=model, kind=kind, perm=p, run=run, S11=ev(sT), S00=ev(s0), cells={})
    for k in ks:
        pre = f"blocks.{k}."
        s01 = {n: (s0[n] if n.startswith(pre) else v) for n, v in sT.items()}
        s10 = {n: (sT[n] if n.startswith(pre) else v) for n, v in s0.items()}
        r["cells"][k] = dict(src=D.KEEP[run][k], S01=ev(s01), S10=ev(s10))
        print(model, kind, p, k, json.dumps(r["cells"][k])[:300], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(f"{OUT}/p61_{model}_{kind}_{p}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
