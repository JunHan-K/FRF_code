"""PREREG54: in-distribution fit of each recovered network re-measured on the fresh set.
Per network: D.evaluate (the training script's own eval) on fresh (all) -> KL to the teacher and top-1 agreement with the
teacher's argmax. Networks = those with a p53 output (same loading as p53.py). Known answer: native network (kind X) -> KL 0.
One job = one model, all its networks; existing outputs are skipped.
usage: p54f.py <model>"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p11  # noqa: E402
import s12r  # noqa: E402
from p47 import e_paths  # noqa: E402

OUT = "out/p54f"


@torch.no_grad()
def main(model):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    nat = S.load(D.M).eval()
    nrm = S.Norm(nat).to(S.dev)
    x, _ = S.cache("fresh")
    lp_t = torch.cat([nat(nrm(x[s:s + 128].to(S.dev).float().div_(255))).float()
                      for s in range(0, len(x), 128)]).log_softmax(-1)
    for f in sorted(glob.glob(f"out/p53/p53_{model}_*.json")):
        d = json.load(open(f))
        kind, p = d["kind"], d["perm"]
        fn = f"{OUT}/p54f_{model}_{kind}_{p}.json"
        if os.path.exists(fn):
            continue
        if kind == "L":
            run, pt, _ = p11.load_case(model, p)
        elif kind == "X":
            run, pt = "D_X", None
            D.KEEP[run] = list(range(12))
        else:
            run, pt, _ = e_paths(model, p, kind)
        m = D.build(run).eval()
        if pt:
            m.load_state_dict(torch.load(pt, map_location=S.dev))
        agree, kl = D.evaluate(m, nrm, x, lp_t.argmax(1).cpu(), lp_t)
        json.dump(dict(model=model, kind=kind, perm=p, run=run, fresh=dict(kl=kl, agree=agree, n=len(x))), open(fn, "w"))
        print(model, kind, p, round(kl, 4), round(agree, 4), flush=True)
        del m


if __name__ == "__main__":
    main(*sys.argv[1:])
