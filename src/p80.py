"""PREREG80 (exploratory follow-up of PREREG79): early probe. Regenerate the step-100 network of the PREREG51 full recovery
(same recipe and RNG order as deit_perm.main: torch.manual_seed(0), teacher, student, data generator seed 0, schedule over
3,000 steps, stop at step 100) and measure, for every PREREG79 cell, the probe baselines:
  squared-gradient norms of blocks.k, the nb block (k+1, or k-1 for k = 11) and the rest (8 fixed id_fit batches);
  skip and swap KL (fresh[:1000]); relative weight change of blocks.k and of the nb block between step 0 and step 100.
FRF at the probe comes from the stored PREREG51 step-100 Grams (out/p57ck/*_p51ck100.npz), not from this job.
KA (exit 1): regenerated step-100 id_hold KL within 3% (relative) of the PREREG51 record; agreement of the regenerated
  block k inserted in the native network with the stored step-100 prediction (fresh, column 13) >= 0.97.
usage: p80.py <worker> <n_workers>"""
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
import numpy as np  # noqa: E402
from p79 import step0, cells, nb_of, select, kl_to, MODELS, KINDS, PERMS, P51  # noqa: E402
from p60 import evaluate  # noqa: E402

OUT = "out/p80"
PROBE = 100
dev = S.dev


def probe_net(model, kind, p):
    D.M = s12r.MODELS[model]
    torch.manual_seed(S.SEED)
    teacher = S.load(D.M).eval().to(dev)
    nrm = S.Norm(teacher).to(dev)
    run, m = step0(model, kind, p)
    s0 = {n: t.detach().clone() for n, t in m.state_dict().items()}
    m.train()
    ps = list(m.parameters())
    opt = torch.optim.AdamW(ps, lr=S.LR, weight_decay=S.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, S.STEPS) / S.STEPS)))
    xf, _ = S.cache("id_fit")
    g = torch.Generator().manual_seed(S.SEED)
    for t in range(1, PROBE + 1):
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
    for q in teacher.parameters():
        q.requires_grad_(False)
    return run, m, s0, teacher, nrm


def main(w, nw):
    os.makedirs(OUT, exist_ok=True)
    data = {n: S.cache(n) for n in ("id_hold", "fresh")}
    jobs = [(m, kd, p) for m in MODELS for kd in KINDS for p in PERMS]
    for i, (model, kind, p) in enumerate(jobs):
        if i % nw != w:
            continue
        fn = f"{OUT}/p80_{model}_{kind}_{p}.json"
        if os.path.exists(fn):
            continue
        t0 = time.time()
        run, m, s0, teacher, nrm = probe_net(model, kind, p)
        r = dict(model=model, kind=kind, perm=p, run=run, cells={})
        r["kl_hold"] = evaluate(m, teacher, nrm, {"id_hold": data["id_hold"]})["id_hold"]["kl"]
        rec = [c for c in json.load(open(f"{P51}/{model}/{run}.json"))["ckpt"] if c["step"] == PROBE][0]
        r["ka_ref"] = rec["kl_teacher"]
        if abs(r["kl_hold"] / r["ka_ref"] - 1) > 0.03:
            print("KA FAIL kl", model, kind, p, r["kl_hold"], r["ka_ref"], flush=True)
            sys.exit(1)
        xf, _ = S.cache("id_fit")
        x1k = data["fresh"][0][:1000]
        sd = m.state_dict()
        for k in cells(model, kind, p):
            st = np.load(f"out/p57ck/p57_{model}_{kind}_{p}_{k}_p51ck{PROBE}.npz")
            pred_st = st["fresh/pri/pred"][:, 13]
            # regenerated block k inserted in the native network (pri context prediction)
            nat = S.load(D.M).eval().to(dev)
            nat.blocks[k] = m.blocks[k]
            m.eval()
            with torch.no_grad():
                pr = torch.cat([nat(nrm(data["fresh"][0][s:s + 128].to(dev).float().div_(255))).argmax(-1).cpu()
                                for s in range(0, len(pred_st), 128)]).numpy()
            agree = float((pr == pred_st).mean())
            if agree < 0.97:
                print("KA FAIL pred", model, kind, p, k, agree, flush=True)
                sys.exit(1)
            del nat
            gq = torch.Generator().manual_seed(790)
            sq = {"blk": 0.0, "nb": 0.0, "rest": 0.0}
            for _ in range(8):
                idx = torch.randint(len(xf), (64,), generator=gq)
                x = nrm(xf[idx].to(dev).float().div_(255))
                m.zero_grad(set_to_none=True)
                lps = m(x).log_softmax(-1)
                with torch.no_grad():
                    lpt = teacher(x).log_softmax(-1)
                ((lpt.exp() * (lpt - lps)).sum(-1).mean()).backward()
                for n, q in m.named_parameters():
                    if q.grad is not None:
                        v = float(q.grad.pow(2).sum())
                        for s in sq:
                            sq[s] += v if select(n, k, s) else 0.0
            m.zero_grad(set_to_none=True)

            def dw(pre):
                a = sum(float((sd[n].float() - s0[n].float()).pow(2).sum()) for n in sd if n.startswith(pre))
                b = sum(float(s0[n].float().pow(2).sum()) for n in s0 if n.startswith(pre))
                return math.sqrt(a / b)
            blk = m.blocks[k]
            c = dict(agree_ka=agree, sq=sq, dw_blk=dw(f"blocks.{k}."), dw_nb=dw(f"blocks.{nb_of(k)}."),
                     kl_probe=kl_to(m, teacher, nrm, x1k))
            m.blocks[k] = torch.nn.Identity()
            c["kl_skip"] = kl_to(m, teacher, nrm, x1k)
            m.blocks[k] = teacher.blocks[k]
            c["kl_swap"] = kl_to(m, teacher, nrm, x1k)
            m.blocks[k] = blk
            r["cells"][k] = c
        json.dump(r, open(fn, "w"), indent=1)
        print(i, model, kind, p, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
