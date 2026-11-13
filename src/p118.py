"""Runner for PREREG117 / 118 / 119 (dogs). For one (model, arm): build the arm (unl118.build), record dog / retained top-1 on
id_hold and fresh, FRF on id_hold forget vs retained images (PREREG118 / 119 only; p110.frf), then the strong attack sftall
(p106.strong_ft, lr 1e-4, 1,000 steps, all parameters, all dog id_fit images, seed 0) and dog / retained top-1 on fresh.
Written 2026-11-11, before any PREREG117-119 value exists.
usage: p118.py <prereg 117|118|119> <model> <arm>"""
import json
import os
import sys

import torch  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p96 import FORGET  # noqa: E402
from p106 import strong_ft  # noqa: E402
import unl118 as U  # noqa: E402

ATT = int(os.environ.get("U118_ATT", 1000))  # paper: default value (override only for quick tests)


def main(pre, model, arm):
    out = os.environ.get(f"P{pre}_OUT", f"out/p{pre}")
    os.makedirs(out, exist_ok=True)
    fn = f"{out}/p{pre}_{model}_{arm}.json"
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M).to(S.dev))
    nrm = S.Norm(nat).to(S.dev)
    (xf, yf), (xh, yh), (x, y) = U.dsets()
    ff = torch.isin(yf, FORGET)
    xd, yd, xr, yr = xf[ff], yf[ff], xf[~ff], yf[~ff]
    res = dict(prereg=pre, model=model, arm=arm, steps=U.STEPS, attack_steps=ATT, info={},
               native=dict(hold=U.accs(nat, nrm, xh, yh), fresh=U.accs(nat, nrm, x, y)))
    m = U.build(model, arm, nat, nrm, xd, yd, xr, yr, xf, yf, res["info"])
    res["before"] = dict(hold=U.accs(m, nrm, xh, yh), fresh=U.accs(m, nrm, x, y))
    print(model, arm, "before", res["before"], res["info"].get("dampened_fraction"), flush=True)
    json.dump(res, open(fn, "w"), indent=1)
    if pre in ("118", "119"):
        res["frf_hold"] = U.hold_frf(nat, m, nrm, xh, yh)
        print(model, arm, "D_hold", round(res["frf_hold"]["D"], 4), flush=True)
        json.dump(res, open(fn, "w"), indent=1)
    idx = torch.nonzero(ff)[:, 0]
    a = strong_ft(m, nrm, xf[idx], yf[idx], ATT, lr=1e-4)
    res["after"] = U.accs(a, nrm, x, y)
    print(model, arm, "after", res["after"], flush=True)
    res["complete"] = True
    json.dump(res, open(fn, "w"), indent=1)
    print("done", model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
