"""PREREG89: teacher KL on the fresh set of the seed-0 ViT-B L networks (not stored in their PREREG57 meta). Same code as
the p57.py net_fresh block. Written 2026-11-05 20:09 KST, before any ViT-B U value exists.
usage: p89kl.py <model>"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p9lib as P  # noqa: E402
import p11  # noqa: E402

OUT = "out/p89"


@torch.no_grad()
def main(model):
    os.makedirs(OUT, exist_ok=True)
    res = {}
    x, y = S.cache("fresh")
    for p in "cdef":
        run, pt, _ = p11.load_case(model, p)
        nat = P.nofuse(S.load(D.M).eval()).to(S.dev)
        nrm = S.Norm(nat).to(S.dev)
        m = P.nofuse(D.build(run).eval())
        m.load_state_dict(torch.load(pt, map_location="cpu"))
        m = m.to(S.dev).eval()
        kl, ag, ok = 0.0, 0, 0
        for s_ in range(0, len(x), 100):
            xb = nrm(x[s_:s_ + 100].to(S.dev).float().div_(255))
            lt, lm = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
            kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
            ag += int((lt.argmax(-1) == lm.argmax(-1)).sum())
            ok += int((lm.argmax(-1).cpu() == y[s_:s_ + 100]).sum())
        res[f"{model}_L_{p}"] = dict(kl=kl / len(x), agree=ag / len(x), top1=ok / len(x), n=len(x))
        print(model, p, res[f"{model}_L_{p}"], flush=True)
        del m, nat
    json.dump(res, open(f"{OUT}/kl_{model}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
