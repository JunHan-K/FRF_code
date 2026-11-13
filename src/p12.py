"""PREREG12 F2/F3: anatomy of the recovery edit dW = W_T - W_0 on shuffled slots 6..11 (no training).
usage: p12.py anat <model> <perm>[_seed1]
  scale : blocks at W_0 + s*dW (all block params)
  random: dW replaced by a random direction with the same per-tensor Frobenius norm (3 draws)
  rank  : each attn/mlp weight-matrix edit truncated to rank r (affine edits kept), plus spectral energy"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p10  # noqa: E402
import p11  # noqa: E402

OUT = "out/p12"
LATE = range(6, 12)
RANKS = (1, 2, 4, 8, 16, 32, 64)


def anat(model, arg):
    os.makedirs(OUT, exist_ok=True)
    run, pt, a0 = p11.load_case(model, arg)
    src = D.KEEP[run]
    nat = S.load(D.M).eval()
    nrm = S.Norm(nat).to(S.dev)
    xh, yh = S.cache("id_hold")
    m = D.build(run).eval()
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    sdT = {kk: v.clone() for kk, v in m.state_dict().items()}
    aT = p10.top1(m, nrm, xh, yh)
    W0 = {}
    for k in LATE:
        for n, v in nat.blocks[src[k]].state_dict().items():
            W0[f"blocks.{k}.{n}"] = v.clone()
    dW = {kk: sdT[kk] - W0[kk] for kk in W0}

    def ev(edit):
        sd = dict(sdT)
        sd.update({kk: W0[kk] + edit[kk] for kk in W0})
        m.load_state_dict(sd)
        a = p10.top1(m, nrm, xh, yh)
        return dict(top1=a, retained=(a - a0) / (aT - a0))

    r = dict(model=model, case=arg, top1_t0=a0, top1_T=aT, scale={}, random=[], rank={}, energy={})
    for s in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        r["scale"][s] = ev({kk: s * v for kk, v in dW.items()})
        print(model, arg, "scale", s, r["scale"][s], flush=True)
    for d in range(3):
        g = torch.Generator(device="cpu").manual_seed(d)
        rnd = {}
        for kk, v in dW.items():
            z = torch.randn(v.shape, generator=g).to(v.device, v.dtype)
            rnd[kk] = z * (v.norm() / z.norm().clamp_min(1e-12))
        r["random"].append(ev(rnd))
        print(model, arg, "random", d, r["random"][-1], flush=True)
    mats = [kk for kk, v in dW.items() if v.dim() == 2 and not p11.is_affine(kk.split(".", 2)[2])]
    if os.environ.get("P12_LINEAR4"):  # same target set for every architecture (attn.qkv/proj, mlp.fc1/fc2)
        mats = [kk for kk in mats if kk.endswith(("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"))]
    svd = {kk: torch.linalg.svd(dW[kk].float(), full_matrices=False) for kk in mats}
    for rk in RANKS:
        edit = dict(dW)
        en = []
        for kk in mats:
            U, Sv, Vh = svd[kk]
            edit[kk] = ((U[:, :rk] * Sv[:rk]) @ Vh[:rk]).to(dW[kk].dtype)
            en.append(float(Sv[:rk].pow(2).sum() / Sv.pow(2).sum()))
        r["rank"][rk] = ev(edit)
        r["energy"][rk] = sum(en) / len(en)
        print(model, arg, "rank", rk, r["rank"][rk], "energy %.3f" % r["energy"][rk], flush=True)
    json.dump(r, open(f"{OUT}/anat_{model}_{arg}{'_lin4' if os.environ.get('P12_LINEAR4') else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    {"anat": anat}[sys.argv[1]](*sys.argv[2:])
