"""PREREG59 D2: replacement of a recovered block by each native block, inside the recovered network itself.
For every p47 cell (slot k of a seed-0 recovered network m), m with blocks[k] replaced by native B_b (b = 0..11).
Per replacement and evaluation set: disagreement with m's own top-1 (per image stored as the replacement's top-1),
KL(m || replaced), top-1 accuracy. Sets: id_hold (comparison) and, with P59_FRESH=1, fresh (final evaluation).
usage: p59.py <model> <L|E|N> <perm>"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p60 import parent_net  # noqa: E402

OUT = os.environ.get("P59_OUT", "out/p59")  # PREREG65: new parents -> out/p65rep
dev = S.dev


@torch.no_grad()
def main(model, kind, p):
    D.M = s12r.MODELS[model]
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    run, m = parent_net(model, kind, p)
    L = len(m.blocks)
    ks = [k for k in range(6, L) if (D.KEEP[run][k] != k if kind == "L" else D.KEEP[run][k] == k)]
    names = ["id_hold"] + (["fresh"] if os.environ.get("P59_FRESH") else [])
    out = dict(meta=np.array(repr(dict(model=model, kind=kind, perm=p, run=run, slots=ks, src=[D.KEEP[run][k] for k in ks]))))
    for n in names:
        x, y = S.cache(n)
        xs = [nrm(x[s:s + 128].to(dev).float().div_(255)) for s in range(0, len(x), 128)]
        base = [m(xb).float() for xb in xs]
        out[f"{n}/label"] = y.numpy()
        out[f"{n}/pred_m"] = torch.cat([z.argmax(-1) for z in base]).cpu().numpy()
        for k in ks:
            preds, kls = [], []
            own = m.blocks[k]
            for b in range(L):
                m.blocks[k] = nat.blocks[b]  # swap in place, restored below
                pr, kl = [], []
                for xb, zb in zip(xs, base):
                    lz, lb = m(xb).float().log_softmax(-1), zb.log_softmax(-1)
                    pr.append(lz.argmax(-1))
                    kl.append((lb.exp() * (lb - lz)).sum(-1))
                preds.append(torch.cat(pr).cpu())
                kls.append(torch.cat(kl).cpu())
            m.blocks[k] = own
            out[f"{n}/{k}/pred_rep"] = torch.stack(preds, 1).numpy().astype(np.int16)  # (n_img, 12)
            out[f"{n}/{k}/kl_rep"] = torch.stack(kls, 1).numpy().astype(np.float32)
            print(model, kind, p, n, k, flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/p59_{model}_{kind}_{p}.npz", **out)


if __name__ == "__main__":
    main(*sys.argv[1:])
