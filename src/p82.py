"""PREREG82 (exploratory): hidden recovery debt. Two recovery paths from the same damaged step-0 network, matched in
recovered performance, then the same second lesions.
  A (compensation): rest-only repair, blocks.k frozen (PREREG79 'rest' recipe, regenerated with the same seeds, 1,000 steps).
  B (direct repair): every parameter trainable, same recipe (data order, augmentation, 1,000-step schedule); KL to the
     native network on fresh[:1000] checked every 25 steps; stop at the first check with KL_B <= KL_A (matched).
Second lesions on fresh[1000:3000] (images not used for matching): skip block j (identity), j = 0..11, j != k.
  D_j(net) = KL(net || net with block j skipped). Also stored for the native and the step-0 damaged network.
KA (exit 1): A's fresh KL within 2% of the PREREG79 rest record; blocks.k of A unchanged.
Cells: PREREG79 L and N cells at slots 7, 9, 11 (36 cells).
usage: p82.py <worker> <n_workers>"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import s12r  # noqa: E402
from p79 import step0, cells, kl_to, MODELS, PERMS, STEPS  # noqa: E402
from p81 import repair_rest  # noqa: E402
from p60 import evaluate  # noqa: E402

OUT = "out/p82"
P79 = "out/p79"
SLOTS, KINDS, CHECK = (7, 9, 11), "LN", 25
dev = S.dev


def repair_full_matched(model, kind, p, teacher, nrm, x_match, target):
    run, m = step0(model, kind, p)
    torch.manual_seed(0)
    ps = list(m.parameters())
    opt = torch.optim.AdamW(ps, lr=S.LR, weight_decay=S.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, STEPS) / STEPS)))
    xf, _ = S.cache("id_fit")
    g = torch.Generator().manual_seed(0)
    trace = []
    for t in range(1, STEPS + 1):
        m.train()
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
        if t % CHECK == 0:
            kl = kl_to(m, teacher, nrm, x_match)
            trace.append((t, kl))
            if kl <= target:
                return m, t, kl, trace
    return m, None, trace[-1][1], trace


@torch.no_grad()
def lesions(m, k, nrm, x):
    """D_j = KL(m || m with block j skipped), j != k"""
    m.eval()
    out = {}
    for j in range(12):
        if j == k:
            continue
        blk = m.blocks[j]
        m.blocks[j] = torch.nn.Identity()
        kl = 0.0
        for s in range(0, len(x), 128):
            xb = nrm(x[s:s + 128].to(dev).float().div_(255))
            m.blocks[j] = blk
            lm = m(xb).float().log_softmax(-1)
            m.blocks[j] = torch.nn.Identity()
            ll = m(xb).float().log_softmax(-1)
            kl += float((lm.exp() * (lm - ll)).sum(-1).sum())
        m.blocks[j] = blk
        out[j] = kl / len(x)
    return out


def main(w, nw):
    os.makedirs(OUT, exist_ok=True)
    fresh = S.cache("fresh")
    x_match, x_les = fresh[0][:1000], fresh[0][1000:3000]
    jobs = []
    for model in MODELS:
        for kind in KINDS:
            for p in PERMS:
                D.M = s12r.MODELS[model]
                jobs += [(model, kind, p, k) for k in cells(model, kind, p) if k in SLOTS]
    cur = None
    for i, (model, kind, p, k) in enumerate(jobs):
        if i % nw != w:
            continue
        fn = f"{OUT}/p82_{model}_{kind}_{p}_{k}.json"
        if os.path.exists(fn):
            continue
        D.M = s12r.MODELS[model]
        if cur != model:
            teacher = S.load(D.M).eval().to(dev)
            for q in teacher.parameters():
                q.requires_grad_(False)
            nrm = S.Norm(teacher).to(dev)
            nat_les = lesions(S.load(D.M).eval().to(dev), -1, nrm, x_les)
            cur = model
        t0 = time.time()
        run, m0 = step0(model, kind, p)
        blk0 = {n: t.detach().clone() for n, t in m0.blocks[k].state_dict().items()}
        r = dict(model=model, kind=kind, perm=p, slot=k, run=run, native=nat_les, step0=lesions(m0, k, nrm, x_les))
        r["kl_step0_match"] = kl_to(m0, teacher, nrm, x_match)
        del m0
        mA = repair_rest(model, kind, p, k, teacher, nrm)
        for n, t in mA.blocks[k].state_dict().items():
            if not torch.equal(t, blk0[n]):
                print("KA FAIL block k changed", model, kind, p, k, n, flush=True)
                sys.exit(1)
        for q in mA.parameters():
            q.requires_grad_(False)
        klA = evaluate(mA, teacher, nrm, {"fresh": fresh})["fresh"]["kl"]
        ref = json.load(open(f"{P79}/p79_{model}_{kind}_{p}_{k}_rest.json"))["eval"]["fresh"]["kl"]
        if abs(klA / ref - 1) > 0.02:
            print("KA FAIL rest kl", model, kind, p, k, klA, ref, flush=True)
            sys.exit(1)
        r["A"] = dict(kl_fresh=klA, kl_match=kl_to(mA, teacher, nrm, x_match), kl_les=kl_to(mA, teacher, nrm, x_les),
                      lesions=lesions(mA, k, nrm, x_les))
        del mA
        torch.cuda.empty_cache()
        mB, tB, klB, trace = repair_full_matched(model, kind, p, teacher, nrm, x_match, r["A"]["kl_match"])
        for q in mB.parameters():
            q.requires_grad_(False)
        r["B"] = dict(step=tB, kl_match=klB, kl_les=kl_to(mB, teacher, nrm, x_les), trace=trace,
                      lesions=lesions(mB, k, nrm, x_les))
        del mB
        torch.cuda.empty_cache()
        json.dump(r, open(fn, "w"), indent=1)
        print(i, model, kind, p, k, "B step", tB, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
