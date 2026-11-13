"""PREREG81 (exploratory): contextual accommodation. Does rest-only repair (blocks.k frozen, every other parameter trained;
the PREREG79 'rest' recipe, regenerated with the same seeds) move the effect of the frozen candidate block u toward the
effect of the native slot block k, selectively relative to control pairs?
Effects (PREREG57 definition): e_b = z(block b at slot k) - z(skip), prefix and suffix of the network under study
(damaged step-0 network, then rest-repaired network), for the 12 native blocks (references) and u = blocks.k of that
network (L: the source block i; N: the noisy block k). Evaluation images id_hold[:500] in both states.
Metrics (inner products summed over images): 'teacher' Fisher (p = native network prediction; identical in both states,
primary), 'own' Fisher (p = the network's own prediction), 'euclid' (centred, uniform p).
KA (exit 1): L step 0, distance between u and the source reference <= 1e-10 (relative) in every metric;
  blocks.k unchanged by rest repair (exact); regenerated rest repair fresh KL within 2% of the PREREG79 record.
usage: p81.py <worker> <n_workers>"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import s12r  # noqa: E402
import e13  # noqa: E402
import spec  # noqa: E402
import p9lib as P  # noqa: E402
import math  # noqa: E402
from p79 import step0, cells, select, MODELS, PERMS, STEPS  # noqa: E402
from p60 import evaluate  # noqa: E402

OUT = "out/p81"
P79 = "out/p79"
KINDS = "LN"
MB = 25
dev = S.dev


def repair_rest(model, kind, p, k, teacher, nrm):
    """copy of p79.repair for strat = 'rest' that returns the network"""
    run, m = step0(model, kind, p)
    torch.manual_seed(0)
    named = list(m.named_parameters())
    for n, q in named:
        q.requires_grad_(select(n, k, "rest"))
    ps = [q for n, q in named if select(n, k, "rest")]
    opt = torch.optim.AdamW(ps, lr=S.LR, weight_decay=S.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, STEPS) / STEPS)))
    xf, _ = S.cache("id_fit")
    g = torch.Generator().manual_seed(0)
    m.train()
    for t in range(1, STEPS + 1):
        idx = torch.randint(len(xf), (S.BS,), generator=g)
        x = nrm(S.augment(xf[idx].to(dev, non_blocking=True), g))
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            lps = m(x[s:s + S.MB]).log_softmax(-1)
            with torch.no_grad():
                lpt = teacher(x[s:s + S.MB]).log_softmax(-1)
            ((lpt.exp() * (lpt - lps)).sum(-1).sum() / S.BS).backward()
        opt.step()
        sched.step()
    return m


@torch.no_grad()
def measure(m, nat, nrm, k, x):
    """summed 13 x 13 Gram (12 native references, then u = m.blocks[k]) per metric, float64"""
    m, nat = P.nofuse(m.eval()), P.nofuse(nat.eval())
    out = {nm: torch.zeros(13, 13, dtype=torch.float64) for nm in ("teacher", "own", "euclid")}
    for s in range(0, len(x), MB):
        xb = nrm(x[s:s + MB].to(dev).float().div_(255))
        h = e13.emb(m, xb)
        for b in m.blocks[:k]:
            h = b(h)

        def suf(o):
            for b in m.blocks[k + 1:]:
                o = b(o)
            return m.forward_head(m.norm(o)).float().double()
        zs = suf(h)
        Z = torch.stack([suf(nat.blocks[b](h)) for b in range(12)] + [suf(m.blocks[k](h))], 1)
        E = Z - zs[:, None]
        pt = nat(xb).float().double().softmax(-1)[:, None]
        po = Z[:, 12].softmax(-1)[:, None]
        pe = torch.full_like(po, 1.0 / po.shape[-1])
        for nm, pp in (("teacher", pt), ("own", po), ("euclid", pe)):
            out[nm] += spec.fip(E, E, pp).sum(0).cpu()
    return {nm: v.numpy() for nm, v in out.items()}


def main(w, nw):
    os.makedirs(OUT, exist_ok=True)
    data = {n: S.cache(n) for n in ("id_hold", "fresh")}
    xh = data["id_hold"][0][:500]
    jobs = []
    for model in MODELS:
        for kind in KINDS:
            for p in PERMS:
                D.M = s12r.MODELS[model]
                jobs += [(model, kind, p, k) for k in cells(model, kind, p)]
    cur = None
    for i, (model, kind, p, k) in enumerate(jobs):
        if i % nw != w:
            continue
        fn = f"{OUT}/p81_{model}_{kind}_{p}_{k}.npz"
        if os.path.exists(fn):
            continue
        D.M = s12r.MODELS[model]
        if cur != model:
            teacher = S.load(D.M).eval().to(dev)
            for q in teacher.parameters():
                q.requires_grad_(False)
            nrm = S.Norm(teacher).to(dev)
            nat = P.nofuse(S.load(D.M).eval().to(dev))
            for q in nat.parameters():
                q.requires_grad_(False)
            cur = model
        t0 = time.time()
        run, m0 = step0(model, kind, p)
        src = D.KEEP[run][k] if kind == "L" else k
        blk0 = {n: t.detach().clone() for n, t in m0.blocks[k].state_dict().items()}
        G0 = measure(m0, nat, nrm, k, xh)
        del m0
        if kind == "L":
            for nm, G in G0.items():
                d = G[12, 12] + G[src, src] - 2 * G[12, src]
                if abs(d) > 1e-10 * G[src, src]:
                    print("KA FAIL step0 u != source", model, kind, p, k, nm, d, flush=True)
                    sys.exit(1)
        torch.cuda.empty_cache()
        m = repair_rest(model, kind, p, k, teacher, nrm)
        for n, t in m.blocks[k].state_dict().items():
            if not torch.equal(t, blk0[n]):
                print("KA FAIL block k changed", model, kind, p, k, n, flush=True)
                sys.exit(1)
        for q in m.parameters():
            q.requires_grad_(False)
        kl = evaluate(m, teacher, nrm, {"fresh": data["fresh"]})["fresh"]["kl"]
        ref = json.load(open(f"{P79}/p79_{model}_{kind}_{p}_{k}_rest.json"))["eval"]["fresh"]["kl"]
        if abs(kl / ref - 1) > 0.02:
            print("KA FAIL rest kl", model, kind, p, k, kl, ref, flush=True)
            sys.exit(1)
        G1 = measure(m, nat, nrm, k, xh)
        np.savez(fn, meta=np.array(repr(dict(model=model, kind=kind, perm=p, slot=k, src=src, run=run,
                                               kl_fresh=kl, kl_fresh_p79=ref, n=len(xh)))),
                 **{f"G0_{nm}": v for nm, v in G0.items()}, **{f"G1_{nm}": v for nm, v in G1.items()})
        del m
        torch.cuda.empty_cache()
        print(i, model, kind, p, k, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
