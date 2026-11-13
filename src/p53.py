"""PREREG53 (input-shift control) and PREREG54 (distribution-shift utility): per recovered network.
  kl : D.evaluate (the training script's own eval) on id_hold (all), shift_R[:NS], shift_A[:NS] -> KL to the teacher and
       top-1 agreement with the teacher's argmax.
  delta (cells = the p47 cells, slots 6..11): input shift faced by the block in slot k, source s = KEEP[k]:
       delta = mean_n |h_k^R(x_n) - h_s^nat(x_n)| / |h_s^nat(x_n)|  (Frobenius over tokens x dims), id_fit[:500],
       h_k^R = output of R.blocks[:k], h_s^nat = output of nat.blocks[:s].
       delta_final uses the recovered weights; delta_0 the step-0 network (D.build(run), no training; L and E only).
  known answers: native network -> delta = 0 and KL = 0 (run with kind 'X').
usage: p53.py <model> <L|E|N|X> <perm>[_seed1]"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p11  # noqa: E402
import s12r  # noqa: E402
from p47 import e_paths  # noqa: E402

OUT = "out/p53"
NA, MB, NS = 500, 50, 2000


@torch.no_grad()
def main(model, kind, p):
    os.makedirs(OUT, exist_ok=True)
    D.M = s12r.MODELS[model]
    if kind == "L":
        run, pt, _ = p11.load_case(model, p)
        rj = e13.paths(model, run)[2]
    elif kind == "X":
        run, pt, rj = "D_X", None, None
        D.KEEP[run] = list(range(12))
    else:
        run, pt, rj = e_paths(model, p, kind)
    nat = S.load(D.M).eval()
    m0 = D.build(run).eval()
    m = D.build(run).eval()
    if pt:
        m.load_state_dict(torch.load(pt, map_location=S.dev))
    nrm = S.Norm(nat).to(S.dev)
    res = dict(model=model, kind=kind, perm=p, run=run, kl={}, cells={})
    for name, n in (("id_hold", None), ("shift_R", NS), ("shift_A", NS)):
        x, _ = S.cache(name)
        x = x[:n] if n else x
        lp_t = torch.cat([nat(nrm(x[s:s + 128].to(S.dev).float().div_(255))).float()
                          for s in range(0, len(x), 128)]).log_softmax(-1)
        agree, kl = D.evaluate(m, nrm, x, lp_t.argmax(1).cpu(), lp_t)
        res["kl"][name] = dict(kl=kl, agree=agree, n=len(x))
    if rj:
        res["kl_train_final"] = json.load(open(rj))["ckpt"][-1]["kl_teacher"]
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])

    def states(net, x):
        h = e13.emb(net, x)
        out = [h]
        for b in net.blocks:
            h = b(h)
            out.append(h)
        return out  # out[k] = input of slot k

    ks = [k for k in range(6, 12) if (D.KEEP[run][k] != k if kind == "L" else True)]
    acc = {k: dict(final=0.0, zero=0.0) for k in ks}
    for s in range(0, NA, MB):
        x = xf[s:s + MB].to(S.dev)
        hn, hr, h0 = states(nat, x), states(m, x), states(m0, x)
        for k in ks:
            ref = hn[D.KEEP[run][k]].flatten(1).double()
            rn = ref.norm(dim=1)
            acc[k]["final"] += float(((hr[k].flatten(1).double() - ref).norm(dim=1) / rn).sum())
            acc[k]["zero"] += float(((h0[k].flatten(1).double() - ref).norm(dim=1) / rn).sum())
    for k in ks:
        res["cells"][k] = dict(k=k, s=D.KEEP[run][k], delta_final=acc[k]["final"] / NA,
                               delta_0=(acc[k]["zero"] / NA) if kind in "LEX" else None)
    print(model, kind, p, {n: round(v["kl"], 4) for n, v in res["kl"].items()},
          {k: (round(c["delta_final"], 3), c["delta_0"] and round(c["delta_0"], 3)) for k, c in res["cells"].items()}, flush=True)
    json.dump(res, open(f"{OUT}/p53_{model}_{kind}_{p}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
