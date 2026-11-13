"""PREREG90 (GPU): block recycling. At the k = 2 slots chosen by FRF magnitude (cfeG order of the Sec. 7 runs), the native
block is replaced by a copy of another native block (the donor; base weights shared, so the unique parameter count equals
removal), then the Sec. 7 recovery (rank-4 LoRA on every block, 3,000 steps, feature KD, lr 1e-3, seed 0) is applied.
Donor rules: frf  = the remaining native block whose effect at slot i is closest to the native block i's effect in the Fisher
                    metric (primary context: native prefix and suffix, p = native prediction, id_fit[:500]);
             adj  = the next block (i+1, or i-1 if i+1 is removed or i = 11);
             rand = uniform among remaining blocks (generator seed 90 + i).
After recovery the recycled block at slot i is read in the frame (12 native references, fresh set, primary context): nu, the
coefficients, and the distance to the donor and to the target block i.
Written 2026-11-06 00:09 KST, before any PREREG90 network exists.
usage: p90.py <model> <frf|adj|rand>"""
import copy
import json
import os
import sys

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import frame as FR  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p11  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
import spec  # noqa: E402

OUT = os.environ.get("P90_OUT", "out/p90")
K, RANK, STEPS, MB = 2, 4, int(os.environ.get("P90_STEPS", 3000)), 25  # P90_STEPS / P90_OUT: smoke test only


@torch.no_grad()
def slot_effects(nat, nrm, x, i, extra=None, keep=None):
    """per-image Fisher Gram of the 12 native effects (+ extra blocks) at slot i, primary context; summed over images"""
    G, Gi = 0.0, []
    for s in range(0, len(x), MB):
        xb = nrm(x[s:s + MB].to(S.dev).float().div_(255))
        h = e13.emb(nat, xb)
        for b in nat.blocks[:i]:
            h = b(h)

        def suffix(o):
            for b in nat.blocks[i + 1:]:
                o = b(o)
            return nat.forward_head(nat.norm(o)).float()

        zs = suffix(h).double()
        Z = [suffix(nat.blocks[r](h)) for r in range(12)] + [suffix(b(h)) for b in (extra or [])]
        Z = torch.stack(Z, 1).double()
        E = Z - zs[:, None]
        g = spec.fip(E, E, Z[:, i].softmax(-1)[:, None]).cpu()
        G = G + g.sum(0)
        if keep is not None:
            Gi.append(g.float())
    return G.numpy(), (torch.cat(Gi).numpy() if keep is not None else None)


def frame(G):
    A, b, u = G[:12, :12], G[:12, 12], G[12, 12]
    return FR.fit(A, b)[0], FR.nu(A, b, u)


def main(model, rule):
    os.makedirs(OUT, exist_ok=True)
    ref = json.load(open(f"out/p18/p23_{model}_cfeG_lora1_fix_main.json"))
    slots = sorted(ref["order"][:K])
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M))
    nrm = S.Norm(nat).to(S.dev)
    allowed = [j for j in range(12) if j not in slots]
    xs, _ = S.cache("id_fit")
    xs = xs[:500]
    donors, dist = {}, {}
    for i in slots:
        if rule == "frf":
            G, _ = slot_effects(nat, nrm, xs, i)
            d = {j: float(G[i, i] - 2 * G[i, j] + G[j, j]) for j in allowed}
            donors[i], dist[i] = min(d, key=d.get), d
        elif rule == "adj":
            donors[i] = i + 1 if i + 1 <= 11 and i + 1 not in slots else i - 1
        else:
            g = torch.Generator().manual_seed(90 + i)
            donors[i] = allowed[int(torch.randint(len(allowed), (1,), generator=g))]
    print(model, rule, "slots", slots, "donors", donors, flush=True)
    m = copy.deepcopy(nat)
    for i in slots:
        m.blocks[i] = copy.deepcopy(nat.blocks[donors[i]])
    with torch.no_grad():
        acc0 = {n: p18.top1(m, nrm, *v) for n, v in p18.eval_sets().items()}
    xf, _ = S.cache("id_fit")
    ps = []
    for blk in m.blocks:
        ps += p18.lora_wrap(blk, RANK)
    n_lora_slot = sum(q.numel() for q in ps) // 12
    p18.kd_train(m, nat, nrm, xf, ps, STEPS, 1e-3, seed=0, featkd=True)
    for blk in m.blocks:
        for parent, nme in p11.lora_targets(blk):
            mod = getattr(parent, nme)
            with torch.no_grad():
                mod.lin.weight += mod.B @ mod.A
            setattr(parent, nme, mod.lin)
    m = p18.frozen(m)
    nb = sum(q.numel() for q in nat.blocks[0].parameters())
    res = dict(model=model, rule=rule, slots=slots, donors=donors, frf_dist=dist, recipe=dict(k=K, rank=RANK, steps=STEPS),
               params_native=sum(q.numel() for q in nat.parameters()),
               params_unique=sum(q.numel() for q in nat.parameters()) - K * nb + K * n_lora_slot, acc_before=acc0)
    with torch.no_grad():
        res["acc"] = {n: p18.top1(m, nrm, *v) for n, v in p18.eval_sets().items()}
        x, _ = S.cache("fresh")
        kl = 0.0
        for s_ in range(0, len(x), 100):
            xb = nrm(x[s_:s_ + 100].to(S.dev).float().div_(255))
            lt, lm = nat(xb).float().log_softmax(-1), m(xb).float().log_softmax(-1)
            kl += float((lt.exp() * (lt - lm)).sum(-1).sum())
        res["net_fresh"] = dict(kl=kl / len(x), n=len(x))
    print(model, rule, res["acc"], res["net_fresh"], flush=True)
    res["frame"] = {}
    for i in slots:
        G, Gi = slot_effects(nat, nrm, x, i, extra=[m.blocks[i]], keep=True)
        w, nu = frame(G)
        dd = lambda j: float((G[12, 12] - 2 * G[12, j] + G[j, j]) / G[12, 12])
        res["frame"][i] = dict(nu=float(nu), E=float(G[12, 12] / len(x)), w=w.tolist(), w_donor=float(w[donors[i]]),
                               w_target=float(w[i]), d_donor=dd(donors[i]), d_target=dd(i))
        np.savez(f"{OUT}/p90_{model}_{rule}_{i}.npz", **{"fresh/pri/G_common": Gi},
                 meta=np.array(repr(dict(model=model, rule=rule, slot=i, donor=donors[i], net_fresh=res["net_fresh"]))))
    json.dump(res, open(f"{OUT}/p90_{model}_{rule}.json", "w"), indent=1)
    print("done", model, rule, res["frame"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
