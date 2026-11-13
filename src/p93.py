"""PREREG93 / PREREG94 (GPU): practical disruptions, then the Sec. 7 recovery, then FRF of blocks 6-11 in the native frame.
  quant <model> <bits> <seed> : PREREG93. Every linear weight of blocks 6-11 quantised to <bits> bits (symmetric, uniform,
                                per output channel); recovery with rank-4 LoRA on every block while the quantised base
                                weights stay fixed (QLoRA-like); adapters folded for measurement.
  swap <model> <donor> <seed> : PREREG94. Blocks 7 and 10 of <model> replaced by blocks 7 and 10 of another ViT-S (<donor>,
                                same width and depth); same recovery.
Recovery = Sec. 7 recipe (rank 4, 3,000 steps, feature KD, lr 1e-3, seed). Each block j = 6..11 of the damaged network
(before recovery) and of the recovered network is read at slot j against the 12 native references (primary context,
common Fisher, fresh set); per-image Gram stored as 'fresh/pri/G_common' (index 12 = the block), as PREREG57/84.
Written 2026-11-06 07:27 KST, before any PREREG93/94 network exists.
usage: p93.py quant <model> <bits> <seed> | p93.py swap <model> <donor> <seed>"""
import copy
import json
import os
import sys

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p11  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p90 import slot_effects  # noqa: E402

OUT = os.environ.get("P93_OUT", "out/p93")
RANK, STEPS = 4, int(os.environ.get("P93_STEPS", 3000))  # P93_STEPS / P93_OUT: smoke test only
SLOTS = [int(k) for k in os.environ.get("P93_SLOTS", "6,7,8,9,10,11").split(",")]  # P93_SLOTS: smoke test only
SWAP = (7, 10)


@torch.no_grad()
def quantise(blk, bits):
    qmax = 2 ** (bits - 1) - 1
    for mod in blk.modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            w.copy_((w / s).round().clamp(-qmax, qmax) * s)


@torch.no_grad()
def fresh_kl(m, nat, nrm, x):
    kl = 0.0
    for s_ in range(0, len(x), 100):
        xb = nrm(x[s_:s_ + 100].to(S.dev).float().div_(255))
        lt, lm = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
        kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
    return kl / len(x)


def measure(m, nat, nrm, x, tag, state, res):
    for j in SLOTS:
        _, Gi = slot_effects(nat, nrm, x, j, extra=[m.blocks[j]], keep=True)
        np.savez(f"{OUT}/p93_{tag}_{state}_{j}.npz", **{"fresh/pri/G_common": Gi},
                 meta=np.array(repr(dict(tag=tag, state=state, slot=j, net_fresh=res[state]))))
        print(tag, state, "slot", j, "saved", flush=True)


def main(mode, model, arg, seed):
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    tag = f"{mode}_{model}_{arg}_s{seed}"
    D.M = s12r.MODELS[model]
    torch.manual_seed(seed)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    m = copy.deepcopy(nat)
    if mode == "quant":
        for j in SLOTS:
            quantise(m.blocks[j], int(arg))
    else:
        donor = p18.frozen(S.load(s12r.MODELS[arg]))
        for j in SWAP:
            m.blocks[j] = copy.deepcopy(donor.blocks[j])
        del donor
    m = p18.frozen(m)
    x, _ = S.cache("fresh")
    res = dict(mode=mode, model=model, arg=arg, seed=seed, recipe=dict(rank=RANK, steps=STEPS, featkd=True))
    with torch.no_grad():
        res["acc_damaged"] = {n: p18.top1(m, nrm, *v) for n, v in p18.eval_sets().items()}
    res["damaged"] = dict(kl=fresh_kl(m, nat, nrm, x), n=len(x))
    measure(m, nat, nrm, x, tag, "damaged", res)
    xf, _ = S.cache("id_fit")
    ps = []
    for blk in m.blocks:
        ps += p18.lora_wrap(blk, RANK)
    p18.kd_train(m, nat, nrm, xf, ps, STEPS, 1e-3, seed=seed, featkd=True)
    for blk in m.blocks:
        for parent, nme in p11.lora_targets(blk):
            mod = getattr(parent, nme)
            with torch.no_grad():
                mod.lin.weight += mod.B @ mod.A
            setattr(parent, nme, mod.lin)
    m = p18.frozen(m)
    with torch.no_grad():
        res["acc"] = {n: p18.top1(m, nrm, *v) for n, v in p18.eval_sets().items()}
    res["recovered"] = dict(kl=fresh_kl(m, nat, nrm, x), n=len(x))
    print(tag, "damaged", res["damaged"], "recovered", res["recovered"], res["acc"], flush=True)
    measure(m, nat, nrm, x, tag, "recovered", res)
    json.dump(res, open(f"{OUT}/p93_{tag}.json", "w"), indent=1)
    print("done", tag, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
