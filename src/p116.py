"""PREREG116 (GPU): recovery after block removal with adapters only next to the removed blocks.
FRF finding used (Fig. beyond b / PREREG88): after removal + recovery, the remaining blocks leave the re-training-only range only
next to a removed block. So: is it enough to adapt only those blocks?
ViT-S deit_s / deit3_s / augreg_s; remove the k = 2 (primary) or k = 4 (secondary) blocks chosen by FRF magnitude (stored order,
out/p18/p23_<model>_cfeG_lora1_fix_main.json, as Sec. 6.1 / PREREG88). Recovery recipe of PREREG88 (rank-4 LoRA, 3,000 steps,
feature KD, lr 1e-3, seed 0, folded), on a subset of the remaining blocks:
  all   every remaining block
  nbr   remaining blocks whose native index is adjacent (|t - d| = 1) to a removed block d
  rand  the same number of remaining non-adjacent blocks, drawn at random (torch generator seed 0); skipped if too few
Measured: top-1 on id_hold and fresh, KL to the native network on fresh, number of trainable parameters.
Written 2026-11-11, before any PREREG116 value exists.
usage: p116.py <model> <k> <arm>"""
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
import p11  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402

OUT = os.environ.get("P116_OUT", "out/p116")
RANK, STEPS = 4, int(os.environ.get("P116_STEPS", 3000))


def main(model, k, arm):
    k = int(k)
    os.makedirs(OUT, exist_ok=True)
    ref = json.load(open(f"out/p18/p23_{model}_cfeG_lora1_fix_main.json"))
    drop = sorted(ref["order"][:k])
    kept = [t for t in range(12) if t not in drop]
    nbr = [t for t in kept if any(abs(t - d) == 1 for d in drop)]
    far = [t for t in kept if t not in nbr]
    if arm == "all":
        sel = kept
    elif arm == "nbr":
        sel = nbr
    elif arm == "rand":
        if len(far) < len(nbr):
            json.dump(dict(model=model, k=k, arm=arm, drop=drop, skipped="too few non-adjacent blocks", complete=True),
                      open(f"{OUT}/p116_{model}_k{k}_{arm}.json", "w"), indent=1)
            return
        g = torch.Generator().manual_seed(0)
        sel = sorted(far[i] for i in torch.randperm(len(far), generator=g)[:len(nbr)].tolist())
    else:
        raise ValueError(arm)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    m = copy.deepcopy(nat)
    m.blocks = nn.Sequential(*[blk for t, blk in enumerate(m.blocks) if t not in drop])
    xf, _ = S.cache("id_fit")
    ps = []
    for t in sel:
        ps += p18.lora_wrap(m.blocks[kept.index(t)], RANK)
    p18.kd_train(m, nat, nrm, xf, ps, STEPS, 1e-3, seed=0, featkd=True)
    for t in sel:
        blk = m.blocks[kept.index(t)]
        for parent, nme in p11.lora_targets(blk):
            mod = getattr(parent, nme)
            with torch.no_grad():
                mod.lin.weight += mod.B @ mod.A
            setattr(parent, nme, mod.lin)
    m = p18.frozen(m)
    res = dict(model=model, k=k, arm=arm, drop=drop, adapted=sel, n_trainable=int(sum(q.numel() for q in ps)))
    with torch.no_grad():
        res["acc"] = {n: p18.top1(m, nrm, *v) for n, v in p18.eval_sets().items()}
        x, y = S.cache("fresh")
        kl, ok = 0.0, 0
        for s_ in range(0, len(x), 100):
            xb = nrm(x[s_:s_ + 100].to(S.dev).float().div_(255))
            lt, lm = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
            kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
            ok += int((lm.argmax(-1).cpu() == y[s_:s_ + 100]).sum())
        res["fresh"] = dict(kl=kl / len(x), top1=ok / len(x))
    res["complete"] = True
    print(model, k, arm, sel, res["acc"], res["fresh"], flush=True)
    json.dump(res, open(f"{OUT}/p116_{model}_k{k}_{arm}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
