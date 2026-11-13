"""PREREG110 (GPU): FRF on unlearned networks. Is the repair restoring blocks for the retained classes but not for the forget group?
Networks (dogs, ViT-S deit_s / deit3_s / augreg_s): native (known-answer control), HRr, OURS, RL from the saved PREREG104 fp16
weights (out/p104w), SCRUB retrained here with the PREREG108 recipe (p108.train, seed 0).
For each network U and slot k = 0..11, U's block k is inserted into slot k of the frozen native network (native input state,
native later blocks, native head); references = the twelve native blocks in slot k; origin = skip; metric = Fisher metric of the
native prediction (native block k in slot k); Gram pooled over images (primary context of the paper). Read on two image sets of the
fresh split: all forget-group images (F) and an equal number of retained-class images (R, seed 0).
Per slot and set: nu = 1 - g' G_nn^+ g / G_BB (out-of-frame share), energy ratio G_BB / G_kk, residual energy nu * G_BB / G_kk.
Written 2026-11-09, before any PREREG110 value exists.
usage: p110.py <model> <arm>"""
import copy
import json
import os
import sys

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
import spec  # noqa: E402
from p91 import acc  # noqa: E402
from p96 import FORGET  # noqa: E402

OUT = os.environ.get("P110_OUT", "out/p110")
W = "out/p104w"
MB = 25


class Skip(nn.Module):
    def forward(self, x, *a, **k):
        return x


def load_arm(model, arm, nat, nrm):
    """returns the unlearned network (frozen, on device)"""
    if arm == "native":
        return copy.deepcopy(nat)
    if arm == "SCRUB":
        import p108
        xf, yf = S.cache("id_fit")
        ff = torch.isin(yf, FORGET)
        p108.xd_labels = yf[ff]
        m = copy.deepcopy(nat)
        return p108.train(m, nat, nrm, xf[ff], xf[~ff], yf[~ff], "SCRUB", int(os.environ.get("P110_STEPS", 3000)))
    m = S.load(D.M)
    sd = torch.load(f"{W}/{model}_{arm}.pt", map_location="cpu")
    own = m.state_dict()
    m.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()})
    return p18.frozen(m.to(S.dev))


@torch.no_grad()
def frf(nat, U, nrm, x):
    out = []
    orig = list(nat.blocks)
    for k in range(12):
        G = torch.zeros(13, 13, dtype=torch.float64)
        for s in range(0, len(x), MB):
            xb = nrm(x[s:s + MB].to(S.dev).float().div_(255))

            def run(mod):
                nat.blocks[k] = mod
                return nat(xb).double()
            zs = run(Skip())
            Z = torch.stack([run(orig[b]) for b in range(12)], 1)
            zb = run(U.blocks[k])
            nat.blocks[k] = orig[k]
            A = torch.cat([Z - zs[:, None], (zb - zs)[:, None]], 1)
            p = Z[:, k].softmax(-1)[:, None]
            G += spec.fip(A, A, p).sum(0).cpu()
        Gnn, g, gbb = G[:12, :12], G[:12, 12], G[12, 12]
        w = torch.linalg.pinv(Gnn, rtol=1e-10) @ g
        nu = float(1 - (w @ g) / gbb)
        out.append(dict(slot=k, nu=nu, energy=float(gbb / G[k, k]), resid=float(nu * gbb / G[k, k]), w=w.tolist()))
    return out


def main(model, arm):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M).to(S.dev))
    nrm = S.Norm(nat).to(S.dev)
    U = load_arm(model, arm, nat, nrm)
    x, y = S.cache("fresh")
    isf = torch.isin(y, FORGET)
    xF, xR_all, yF, yR_all = x[isf], x[~isf], y[isf], y[~isf]
    g = torch.Generator().manual_seed(0)
    ri = torch.randperm(len(xR_all), generator=g)[:len(xF)]
    res = dict(model=model, arm=arm, nF=len(xF), nR=len(ri),
               final=dict(forget=acc(U, nrm, xF, yF), retain=acc(U, nrm, xR_all, yR_all)))
    print(model, arm, res["final"], flush=True)
    res["F"] = frf(nat, U, nrm, xF)
    print(model, arm, "F nu", [round(r["nu"], 3) for r in res["F"]], flush=True)
    res["R"] = frf(nat, U, nrm, xR_all[ri])
    print(model, arm, "R nu", [round(r["nu"], 3) for r in res["R"]], flush=True)
    json.dump(res, open(f"{OUT}/p110_{model}_{arm}.json", "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
