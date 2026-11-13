"""PREREG10. usage: p10.py rewind <model> <real_p[_seed1]|DC>   |   p10.py traj <model> <perm>"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402

OUT = "out/p10"


@torch.no_grad()
def top1(m, nrm, x, y):
    m.eval()
    p = torch.cat([m(nrm(x[s:s + 128].to(S.dev).float().div_(255))).argmax(1).cpu() for s in range(0, len(x), 128)])
    return float((p == y).float().mean())


def rewound(m, nat, src, blocks):
    sd = {kk: v.clone() for kk, v in m.state_dict().items()}
    for k in blocks:
        for n, v in nat.blocks[src[k]].state_dict().items():
            sd[f"blocks.{k}.{n}"] = v.clone()
    return sd


@torch.no_grad()
def alpha(m, nat, nrm, x, src):
    out = {}
    for k in range(12):
        i, j = src[k], k
        if abs(i - j) < 2:
            continue
        num = 0.0
        for s in range(0, 500, 50):
            h = e13.emb(m, nrm(x[s:s + 50].to(S.dev).float().div_(255)))
            for b in m.blocks[:k]:
                h = b(h)
            d0 = (nat.blocks[i](h) - h).flatten(1)
            t = (nat.blocks[j](h) - h).flatten(1) - d0
            num += float(((m.blocks[k](h) - h).flatten(1) - d0).mul(t).sum(1).div(t.pow(2).sum(1).clamp_min(1e-12)).sum())
        out[k] = num / 500
    return out


def evaluate_case(model, run, pt, nat, nrm, xh, yh, a0):
    src = D.KEEP[run]
    m = D.build(run).eval()
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    aT = top1(m, nrm, xh, yh)
    sdT = {kk: v.clone() for kk, v in m.state_dict().items()}
    r = dict(top1_t0=a0, top1_T=aT)
    for name, blocks in (("rewind_shuffled", range(6, 12)), ("rewind_context", range(0, 6)),
                         ("rewind_sep", [k for k in range(12) if abs(src[k] - k) >= 2])):
        m.load_state_dict(rewound(m, nat, src, blocks))
        a = top1(m, nrm, xh, yh)
        r[name] = dict(top1=a, retained=(a - a0) / (aT - a0) if aT != a0 else float("nan"))
        m.load_state_dict(sdT)
    return m, r


def main(mode, model, arg):
    os.makedirs(OUT, exist_ok=True)
    D.M = e13.MODELS[model]
    nat = S.load(D.M).eval()
    for q in nat.parameters():
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    xh, yh = S.cache("id_hold")
    if mode == "rewind":
        if arg == "DC":
            run = "D_C"
            pt = "out/depth/D_C.pt" if model == "deit_s" else f"out/s12r_{model}/D_C.pt"
            a0 = top1(nat, nrm, xh, yh)
        else:
            parts = arg.split("_")
            e13.SEED_SFX = "_" + parts[2] if len(parts) > 2 else ""
            run = f"D_L{parts[1]}"
            pt, _, rj = e13.paths(model, run)
            a0 = json.load(open(rj))["ckpt"][0]["top1"]
        _, r = evaluate_case(model, run, pt, nat, nrm, xh, yh, a0)
        r.update(model=model, case=arg)
        print(r, flush=True)
        json.dump(r, open(f"{OUT}/rewind_{model}_{arg}.json", "w"), indent=1)
    else:
        run = f"D_L{arg}"
        SFX = os.environ.get("P10_SFX", "")  # PREREG11 replication seeds
        S.OUT = f"{OUT}/traj_{model}{SFX}"
        os.makedirs(S.OUT, exist_ok=True)
        if not os.path.exists(f"{S.OUT}/{run}_ck3000.pt"):
            D.main(run)
        a0 = json.load(open(f"{S.OUT}/{run}.json"))["ckpt"][0]["top1"]
        res = dict(model=model, run=run, ckpts={})
        for t in (0, 100, 300, 1000, 3000):
            m, r = evaluate_case(model, run, f"{S.OUT}/{run}_ck{t}.pt", nat, nrm, xh, yh, a0)
            al = alpha(m, nat, nrm, xh, D.KEEP[run])
            r.update(alpha=al, alpha_median=float(np.median(list(al.values()))),
                     frac_slot=float(np.mean([v >= 0.5 for v in al.values()])))
            res["ckpts"][t] = r
            print(model, run, t, {a: b for a, b in r.items() if a != "alpha"}, flush=True)
        json.dump(res, open(f"{OUT}/traj_{model}_{run}{SFX}.json", "w"), indent=1)
        if SFX:  # disk quota: drop intermediate checkpoints of replication runs
            for t in (0, 100, 300, 1000):
                os.remove(f"{S.OUT}/{run}_ck{t}.pt")


if __name__ == "__main__":
    main(*sys.argv[1:])
