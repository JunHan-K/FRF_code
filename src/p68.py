"""PREREG67 Q4: joint replacement of two blocks of a recovered network m (slots a < b, both cells of the parent).
Policies (coefficients fixed per slot from single-slot calibration, never refitted on joint behaviour):
  single : per slot, the native block with the smallest KL(m || m with B_n at that slot) on calib = id_fit[:500]
  fisher : per slot, the simplex fit of the logit effect in the common Fisher metric (p57 calib/pri/G_common[:500])
Evaluation on fresh, per policy: logits z0 (m), zA (slot a replaced), zB (slot b), zAB (both). KL(m || .) and disagreement
for A, B, AB; interaction I = zAB - zA - zB + z0 with energy E_I = mean ||I||^2 in the Fisher metric of m's prediction, and
the energies of zA - z0, zB - z0 for scale. KL is not additive; KL_AB - KL_A - KL_B is reported but not read as interaction.
usage: p68.py <model> <L|E|N> <perm> <slot_a> <slot_b>"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p60 import parent_net  # noqa: E402
from p66 import simplex_ls, fisher_ip, P57DIRS  # noqa: E402

OUT = os.environ.get("P68_OUT", "out/p68")
dev = S.dev
NC, MB = 500, 50


class Mix(nn.Module):
    def __init__(self, refs, w):
        super().__init__()
        self.refs, self.w = refs, [float(v) for v in w]

    def forward(self, h):
        return h + sum(wb * (r(h) - h) for r, wb in zip(self.refs, self.w) if wb != 0.0)


@torch.no_grad()
def main(model, kind, perm, sa, sb):
    a, b = int(sa), int(sb)
    D.M = s12r.MODELS[model]
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    run, m = parent_net(model, kind, perm)
    L = len(m.blocks)
    refs = [nat.blocks[i] for i in range(L)]
    own = {a: m.blocks[a], b: m.blocks[b]}
    xc, _ = S.cache("id_fit")
    cal = [nrm(xc[s:s + MB].to(dev).float().div_(255)) for s in range(0, NC, MB)]
    base = [m(x).float().log_softmax(-1) for x in cal]
    W = {"single": {}, "fisher": {}}
    for k in (a, b):
        kl = np.zeros(L)
        for i in range(L):
            m.blocks[k] = refs[i]
            kl[i] = sum(float((lb.exp() * (lb - m(x).float().log_softmax(-1))).sum()) for x, lb in zip(cal, base))
        m.blocks[k] = own[k]
        W["single"][k] = np.eye(L)[int(np.argmin(kl))]
        z = next(np.load(f"{d_}/p57_{model}_{kind}_{perm}_{k}.npz") for d_ in P57DIRS
                 if os.path.exists(f"{d_}/p57_{model}_{kind}_{perm}_{k}.npz"))
        G = z["calib/pri/G_common"][:NC].sum(0).astype(np.float64)
        W["fisher"][k] = simplex_ls(G[:L, :L], G[:L, L])
    xf, _ = S.cache("fresh")
    res = dict(model=model, kind=kind, perm=perm, slots=[a, b], w={p: {str(k): [round(float(v), 6) for v in w] for k, w in d.items()}
                                                                  for p, d in W.items()})
    for pol, wd in W.items():
        acc = dict(kl_A=0.0, kl_B=0.0, kl_AB=0.0, dis_A=0, dis_B=0, dis_AB=0, E_I=0.0, E_A=0.0, E_B=0.0)
        for s in range(0, len(xf), MB):
            x = nrm(xf[s:s + MB].to(dev).float().div_(255))
            z0 = m(x).double()
            m.blocks[a] = Mix(refs, wd[a])
            zA = m(x).double()
            m.blocks[b] = Mix(refs, wd[b])
            zAB = m(x).double()
            m.blocks[a] = own[a]
            zB = m(x).double()
            m.blocks[b] = own[b]
            l0, p = z0.log_softmax(-1), z0.softmax(-1)
            for tag, zz in (("A", zA), ("B", zB), ("AB", zAB)):
                lz = zz.log_softmax(-1)
                acc[f"kl_{tag}"] += float((p * (l0 - lz)).sum())
                acc[f"dis_{tag}"] += int((lz.argmax(-1) != l0.argmax(-1)).sum())
            I = zAB - zA - zB + z0
            acc["E_I"] += float(fisher_ip(I, I, p).sum())
            acc["E_A"] += float(fisher_ip(zA - z0, zA - z0, p).sum())
            acc["E_B"] += float(fisher_ip(zB - z0, zB - z0, p).sum())
        res[pol] = {q: v / len(xf) for q, v in acc.items()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/p68_{model}_{kind}_{perm}_{a}_{b}.json", "w"), indent=1)
    print(json.dumps(res)[:1500], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
