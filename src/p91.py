"""PREREG91 (GPU): unlearned at the output, erased in the blocks? Forget set = the 100 ImageNet classes c with c % 10 == 0.
Unlearning: full-parameter distillation (the recovery recipe of the paper: 3,000 steps, batch 64, lr S.LR, wd S.WD, seed 0)
from the native network with the forget-class logits masked (-1e4), on id_fit (all images, no labels used).
Measured on fresh (true labels; forget = label in F):
  acc        native / unlearned top-1 on forget and retain images;
  stitch(k)  unlearned embedding + unlearned blocks[:k] + native blocks[k:] + native norm and head, k = 0..12 (k = 12: only
             the norm and head are native): forget / retain top-1 -> how deep the forgetting is written;
  relearn    rank-4 LoRA on the unlearned network, distillation from the native network on retain-class id_fit images only
             (forget images never seen), 300 steps, lr 1e-3: forget top-1 afterwards;
  frame      every unlearned block k read at slot k in the native frame (12 native references, primary context, common Fisher),
             separately on forget images and on retain images[:500]: nu and E, and the energy of the change |e_u - e_k|^2.
Written 2026-11-06 00:09 KST, before any PREREG91 network exists.
usage: p91.py <model>"""
import copy
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p90 import slot_effects, frame  # noqa: E402

OUT = os.environ.get("P91_OUT", "out/p91")
STEPS = int(os.environ.get("P91_STEPS", 3000))  # P91_STEPS / P91_OUT: smoke test only
RSTEPS = int(os.environ.get("P91_RSTEPS", 300))
FORGET = torch.arange(0, 1000, 10)


class Masked(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        z = self.m(x).float().clone()
        z[:, FORGET.to(z.device)] = -1e4
        return z


@torch.no_grad()
def stitched_acc(stu, nat, nrm, x, y, k, bs=100):
    ok = 0
    for s in range(0, len(x), bs):
        h = e13.emb(stu, nrm(x[s:s + bs].to(S.dev).float().div_(255)))
        for b in stu.blocks[:k]:
            h = b(h)
        for b in nat.blocks[k:]:
            h = b(h)
        ok += int((nat.forward_head(nat.norm(h)).argmax(-1).cpu() == y[s:s + bs]).sum())
    return ok / max(len(x), 1)


@torch.no_grad()
def acc(m, nrm, x, y, bs=100):
    ok = 0
    for s in range(0, len(x), bs):
        ok += int((m(nrm(x[s:s + bs].to(S.dev).float().div_(255))).argmax(-1).cpu() == y[s:s + bs]).sum())
    return ok / max(len(x), 1)


def main(model):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    x, y = S.cache("fresh")
    isf = torch.isin(y, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    res = dict(model=model, n_forget=int(isf.sum()), n_retain=int((~isf).sum()), recipe=dict(steps=STEPS, relearn=RSTEPS))
    res["native"] = dict(forget=acc(nat, nrm, xF, yF), retain=acc(nat, nrm, xR, yR))
    pt = f"{OUT}/net_{model}.pt"
    stu = copy.deepcopy(nat)
    if os.path.exists(pt):
        stu.load_state_dict(torch.load(pt, map_location=S.dev))
    else:
        xf, _ = S.cache("id_fit")
        p18.kd_train(stu, Masked(nat), nrm, xf, list(stu.parameters()), STEPS, S.LR, S.WD, seed=0)
        torch.save(stu.state_dict(), pt)
    stu = p18.frozen(stu)
    res["unlearned"] = dict(forget=acc(stu, nrm, xF, yF), retain=acc(stu, nrm, xR, yR))
    print(model, "native", res["native"], "unlearned", res["unlearned"], flush=True)
    res["stitch"] = {k: dict(forget=stitched_acc(stu, nat, nrm, xF, yF, k), retain=stitched_acc(stu, nat, nrm, xR, yR, k))
                     for k in range(13)}
    print(model, "stitch", {k: round(v["forget"], 3) for k, v in res["stitch"].items()}, flush=True)
    # relearning from retain-class images only
    rl = copy.deepcopy(stu)
    xf, yf = S.cache("id_fit")
    xr = xf[~torch.isin(yf, FORGET)]
    ps = []
    for blk in rl.blocks:
        ps += p18.lora_wrap(blk, 4)
    p18.kd_train(rl, nat, nrm, xr, ps, RSTEPS, 1e-3, seed=0)
    rl = p18.frozen(rl)
    res["relearn"] = dict(forget=acc(rl, nrm, xF, yF), retain=acc(rl, nrm, xR, yR))
    print(model, "relearn", res["relearn"], flush=True)
    del rl
    res["frame"] = {}
    for k in range(12):
        r = {}
        for name, xs in (("forget", xF), ("retain", xR[:500])):
            G, _ = slot_effects(nat, nrm, xs, k, extra=[stu.blocks[k]])
            w, nu = frame(G)
            r[name] = dict(nu=float(nu), E=float(G[12, 12] / len(xs)),
                           change=float((G[12, 12] - 2 * G[12, k] + G[k, k]) / len(xs)), w_self=float(w[k]))
        res["frame"][k] = r
        print(model, "slot", k, r, flush=True)
    # known answer: stitch with nothing from the unlearned network except the embedding (k = 0) vs the native network
    res["KA"] = dict(unlearned=res["unlearned"]["forget"] <= 0.1 * res["native"]["forget"])
    json.dump(res, open(f"{OUT}/p91_{model}.json", "w"), indent=1)
    print("done", model, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
