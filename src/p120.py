"""PREREG120 (GPU): the dog results PREREG110 / 112 / 114 / 115 / 118 repeated on the other forget groups of PREREG108
(birds 59, reptiles 36, insects 27 classes), ViT-S deit_s / deit3_s / augreg_s. No weights are saved (disk quota): every
network is built (unl118.build, seed 0) and measured in the same run.
Arms (per group and model):
  full arms   native, HRr, RL, SCRUB, and OURS = SDa<alpha> with the PREREG108-selected alpha (3 or 10, see OURS_ALPHA)
  selection   HRr, SDa100, SDa30, SDa10, SDa3, SDa1.5, SD4a3 (PREREG118 configurations; the OURS one is the OURS run)
Per run (fresh split unless noted; group = forget classes):
  all arms        fresh / id_hold group and retained top-1 before; D_hold = PREREG118 hold FRF (not for native);
                  forget-only attack sftall (p106.strong_ft, lr 1e-4, 1,000 steps, all group id_fit images) -> top-1 after
  full arms       fresh FRF before (PREREG110) and after that attack (PREREG114)
  full arms \\ native  mixed attacks (p112.mixed_ft, lr 1e-4 and 3e-4, 1,000 steps) -> top-1 after (PREREG112);
                  block rewind with native group head rows: base, single(k), deep(j) (PREREG115)
Written 2026-11-11, before any PREREG120 value exists.
usage: p120.py <group> <model> <arm>    (arm = native | HRr | RL | SCRUB | OURS | SDa100 | SDa30 | SDa10 | SDa3 | SDa1.5 | SD4a3)"""
import copy
import json
import os
import sys

import torch  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import p96  # noqa: E402

GROUPS = {"birds": list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),
          "reptiles": list(range(33, 69)), "insects": list(range(300, 327))}
GROUP = sys.argv[1] if __name__ == "__main__" else "birds"
p96.FORGET = torch.tensor(GROUPS[GROUP])  # before any module that binds FORGET at import
FORGET = p96.FORGET
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import p18  # noqa: E402
import s12r  # noqa: E402
from p91 import acc  # noqa: E402
from p106 import strong_ft  # noqa: E402
from p110 import frf  # noqa: E402
from p112 import mixed_ft  # noqa: E402
import unl118 as U  # noqa: E402
import select_ours as SEL  # noqa: E402

assert U.FORGET is FORGET
OUT = os.environ.get("P120_OUT", "out/p120")
ATT = int(os.environ.get("P120_ATT", 1000))  # paper: default value (override only for quick tests)
# damage strength of OURS, chosen on id_hold only (PREREG102 rule on the p108 runs; see src/select_ours.py)
OURS_ALPHA = {g: SEL.SELECTED[g] for g in GROUPS}
FULL = ("native", "HRr", "RL", "SCRUB", "OURS")


@torch.no_grad()
def rewind(nat, m, nrm, xF, yF, xR, yR):
    Um = copy.deepcopy(m)
    f = FORGET.to(S.dev)
    Um.head.weight[f] = nat.head.weight[f].to(Um.head.weight)
    Um.head.bias[f] = nat.head.bias[f].to(Um.head.bias)
    own = list(Um.blocks)

    def ev(rew):
        for k in range(12):
            Um.blocks[k] = nat.blocks[k] if k in rew else own[k]
        r = dict(forget=acc(Um, nrm, xF, yF), retain=acc(Um, nrm, xR, yR))
        for k in range(12):
            Um.blocks[k] = own[k]
        return r
    out = dict(base=ev(set()), single={k: ev({k}) for k in range(12)}, deep={j: ev(set(range(j, 12))) for j in range(11, -1, -1)})
    return out


def main(group, model, arm):
    os.makedirs(OUT, exist_ok=True)
    fn = f"{OUT}/p120_{group}_{model}_{arm}.json"
    if arm == "OURS":  # re-derive the selection from this model's p108 runs if they are present
        if SEL.available(group, models=(model,)):
            a = SEL.select(group, model)
            if a != OURS_ALPHA[group][model]:
                sys.exit(f"OURS alpha from out/p108 ({a}) differs from select_ours.SELECTED ({OURS_ALPHA[group][model]})")
            print(f"OURS alpha {a} re-derived from out/p108", flush=True)
        else:
            print(f"OURS alpha {OURS_ALPHA[group][model]} from select_ours.SELECTED (out/p108 runs for {group} {model} "
                  "not present, not re-derived)", flush=True)
    D.M = s12r.MODELS[model]
    torch.manual_seed(0)
    nat = p18.frozen(S.load(D.M).to(S.dev))
    nrm = S.Norm(nat).to(S.dev)
    (xf, yf), (xh, yh), (x, y) = U.dsets()
    ff = torch.isin(yf, FORGET)
    xd, yd, xr, yr = xf[ff], yf[ff], xf[~ff], yf[~ff]
    isf = torch.isin(y, FORGET)
    xF, yF, xR, yR = x[isf], y[isf], x[~isf], y[~isf]
    ri = torch.randperm(len(xR), generator=torch.Generator().manual_seed(0))[:len(xF)]
    spec = f"SDa{OURS_ALPHA[group][model]}" if arm == "OURS" else arm
    res = dict(prereg=120, group=group, model=model, arm=arm, spec=spec, steps=U.STEPS, attack_steps=ATT, nF=len(xF), info={},
               native=dict(hold=U.accs(nat, nrm, xh, yh), fresh=U.accs(nat, nrm, x, y)))
    torch.manual_seed(0)
    m = U.build(model, spec, nat, nrm, xd, yd, xr, yr, xf, yf, res["info"])
    m = p18.frozen(m)
    res["before"] = dict(hold=U.accs(m, nrm, xh, yh), fresh=U.accs(m, nrm, x, y))
    print(group, model, arm, spec, "before", res["before"], res["info"].get("dampened_fraction"), flush=True)
    json.dump(res, open(fn, "w"), indent=1)
    if arm != "native":
        res["frf_hold"] = U.hold_frf(nat, m, nrm, xh, yh)
        print(group, model, arm, "D_hold", round(res["frf_hold"]["D"], 4), flush=True)
        json.dump(res, open(fn, "w"), indent=1)
    full = arm in FULL
    if full:
        res["pre"] = dict(F=frf(nat, m, nrm, xF), R=frf(nat, m, nrm, xR[ri]))
        res["pre"]["D"] = U.Dstat(res["pre"])
        print(group, model, arm, "D_pre", round(res["pre"]["D"], 4), flush=True)
        json.dump(res, open(fn, "w"), indent=1)
    idx = torch.nonzero(ff)[:, 0]
    a = strong_ft(m, nrm, xf[idx], yf[idx], ATT, lr=1e-4)
    res["after"] = U.accs(a, nrm, x, y)
    print(group, model, arm, "after", res["after"], flush=True)
    if full:
        res["post"] = dict(F=frf(nat, a, nrm, xF), R=frf(nat, a, nrm, xR[ri]))
        res["post"]["D"] = U.Dstat(res["post"])
        print(group, model, arm, "D_post", round(res["post"]["D"], 4), flush=True)
    del a
    torch.cuda.empty_cache()
    json.dump(res, open(fn, "w"), indent=1)
    if full and arm != "native":
        res["mixed"] = {}
        for lr in (1e-4, 3e-4):
            a = mixed_ft(m, nrm, xd, yd, xr, yr, ATT, lr)
            res["mixed"][f"lr{lr:g}"] = U.accs(a, nrm, x, y)
            print(group, model, arm, "mixed", lr, res["mixed"][f"lr{lr:g}"], flush=True)
            del a
            torch.cuda.empty_cache()
            json.dump(res, open(fn, "w"), indent=1)
        res["rewind"] = rewind(nat, m, nrm, xF, yF, xR, yR)
        print(group, model, arm, "rewind base", res["rewind"]["base"], "deep0", res["rewind"]["deep"][0], flush=True)
    res["complete"] = True
    json.dump(res, open(fn, "w"), indent=1)
    print("done", group, model, arm, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
