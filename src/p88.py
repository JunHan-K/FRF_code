"""PREREG88 (GPU): practical recovery. Remove the k = 2 blocks chosen by FRF magnitude (cfeG order stored in
out/p18/p23_<model>_cfeG_lora1_fix_main.json), recover with the Sec. 7 recipe (rank-4 LoRA on every remaining block,
3,000 steps, feature KD, lr 1e-3, folded), save the network, then read every remaining block j (slots 6-11) at its native
slot against the 12 native references: primary context (native prefix and suffix), common Fisher (p = native prediction),
fresh set; per-image 13x13 Gram stored as 'fresh/pri/G_common' (index 12 = the recovered block), as PREREG57.
Written 2026-11-05 20:09 KST, before any PREREG88 network exists.
KA (exit 1): seed 0 id_hold top-1 within 1.0 point of the stored Sec. 7 run; the native block j read as a candidate
(index 13 of an extra Gram on the first 200 images) has nu <= 1e-4.
usage: p88.py <model> <seed>"""
import copy
import json
import os
import sys

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import frame as FR  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p11  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
import spec  # noqa: E402

OUT = os.environ.get("P88_OUT", "out/p88")
K, RANK, STEPS, MB = 2, 4, int(os.environ.get("P88_STEPS", 3000)), 25  # P88_STEPS / P88_OUT: smoke test only


def nu_of(G):
    return FR.nu_from_gram(G, 12, -1)


def main(model, seed):
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    tag = f"{model}_k{K}_s{seed}"
    ref = json.load(open(f"out/p18/p23_{model}_cfeG_lora1_fix_main.json"))
    drop = sorted(ref["order"][:K])
    D.M = s12r.MODELS[model]
    torch.manual_seed(seed)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    pt = f"{OUT}/net_{tag}.pt"
    m = copy.deepcopy(nat)
    m.blocks = nn.Sequential(*[blk for t, blk in enumerate(m.blocks) if t not in drop])
    if os.path.exists(pt):
        m.load_state_dict(torch.load(pt, map_location=S.dev))
    else:
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
        torch.save(m.state_dict(), pt)
    m = p18.frozen(m)
    res = dict(model=model, seed=seed, drop=drop, recipe=dict(k=K, rank=RANK, steps=STEPS, featkd=True, sel="cfeG"))
    with torch.no_grad():
        res["acc"] = {n: p18.top1(m, nrm, *v) for n, v in p18.eval_sets().items()}
        res["acc_ref_seed0"] = [s for s in ref["steps"] if s["k"] == K][0]["acc"]
        x, y = S.cache("fresh")
        kl = 0.0
        for s_ in range(0, len(x), 100):
            xb = nrm(x[s_:s_ + 100].to(S.dev).float().div_(255))
            lt, lm = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
            kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
        res["net_fresh"] = dict(kl=kl / len(x), n=len(x))
    print(tag, res["acc"], res["net_fresh"], flush=True)
    kept = [t for t in range(12) if t not in drop]
    res["cells"] = {}
    for j in range(6, 12):
        if j in drop:
            continue
        cand = m.blocks[kept.index(j)]
        Gs, Gk = [], 0.0
        with torch.no_grad():
            for s in range(0, len(x), MB):
                xb = nrm(x[s:s + MB].to(S.dev).float().div_(255))
                h = e13.emb(nat, xb)
                for b in nat.blocks[:j]:
                    h = b(h)

                def suffix(o):
                    for b in nat.blocks[j + 1:]:
                        o = b(o)
                    return nat.forward_head(nat.norm(o)).float()

                zs = suffix(h).double()
                Z = torch.stack([suffix(nat.blocks[r](h)) for r in range(12)], 1).double()
                C = torch.stack([suffix(cand(h)), suffix(nat.blocks[j](h))], 1).double()
                E = torch.cat([Z, C], 1) - zs[:, None]
                g = spec.fip(E, E, Z[:, j].softmax(-1)[:, None]).cpu()
                Gs.append(g[:, :13, :13].float())
                if s < 200:
                    Gk = Gk + g[:, [*range(12), 13]][:, :, [*range(12), 13]].sum(0)
        G = torch.cat(Gs).numpy()
        nu_copy = nu_of(Gk.numpy())
        np.savez(f"{OUT}/p88_{tag}_{j}.npz", **{"fresh/pri/G_common": G},
                 meta=np.array(repr(dict(model=model, seed=seed, slot=j, drop=drop, net_fresh=res["net_fresh"], nu_copy=nu_copy))))
        res["cells"][j] = dict(nu_copy=float(nu_copy))
        print(tag, "slot", j, "saved", flush=True)
    d = abs(res["acc"]["id_hold"] - res["acc_ref_seed0"]["id_hold"])
    res["KA"] = dict(top1_diff=d, nu_copy_max=max(c["nu_copy"] for c in res["cells"].values()))
    res["KA"]["pass_"] = bool((seed != 0 or d <= 0.01) and res["KA"]["nu_copy_max"] <= 1e-4)
    json.dump(res, open(f"{OUT}/p88_{tag}.json", "w"), indent=1)
    if not res["KA"]["pass_"]:
        print("KA FAILED", res["KA"], flush=True)
        sys.exit(1)
    print("done", tag, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
