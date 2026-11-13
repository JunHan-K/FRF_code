"""Damage strength of OURS per forget group and model, chosen on id_hold only (PREREG102 rule, applied in PREREG108):
among the damage arms SD3-12H and SD10-12H (selective dampening, alpha 3 / 10, all 12 blocks, head re-init + repair),
keep those with id_hold forget top-1 <= 0.1 x native and id_hold retained top-1 drop <= 5 points, and take the one with
the smallest id_hold retained drop. No fresh-set value enters the choice. Reads the p108 outputs (out/p108).

SELECTED is the table this rule gave in the paper runs; p120 (birds, reptiles, insects) and unl118 (dogs) use it, and p120
re-derives it from out/p108 whenever those files exist and stops on any difference. SELECTED_VITB is the same rule applied
to the ViT-B dog runs of p102 (PREREG103; P102_OUT=out/p103 python src/p102.py <model> <arm>), record "repaired".
usage: python src/select_ours.py [--check] [group ...]      (default: dogs birds reptiles insects)
       --check: exit 1 if the derived table differs from SELECTED (and SELECTED_VITB, if out/p103 exists)"""
import json
import os
import sys

MODELS = ("deit_s", "deit3_s", "augreg_s")
ARMS = {"SD3-12H": 3, "SD10-12H": 10}
SELECTED = {"dogs": {"deit_s": 3, "deit3_s": 10, "augreg_s": 10},
            "birds": {"deit_s": 3, "deit3_s": 10, "augreg_s": 10},
            "reptiles": {"deit_s": 10, "deit3_s": 10, "augreg_s": 10},
            "insects": {"deit_s": 3, "deit3_s": 10, "augreg_s": 10}}
VITB = ("deit_b", "deit3_b", "augreg_b")
SELECTED_VITB = {"deit_b": 10, "deit3_b": 10, "augreg_b": 10}


def rule(nat, finals):
    """nat: native record ({forget_hold, retain_hold}); finals: {alpha: final record of that damage arm}.
    Returns the selected alpha, or None if no arm passes."""
    ok = {}
    for alpha, f in finals.items():
        drop = nat["retain_hold"] - f["retain_hold"]
        if f["forget_hold"] <= 0.1 * nat["forget_hold"] and drop <= 0.05:
            ok[alpha] = drop
    return min(ok, key=ok.get) if ok else None


def select(group, model, out="out/p108"):
    nat = json.load(open(f"{out}/p108_{group}_{model}_native.json"))["native"]
    finals = {alpha: json.load(open(f"{out}/p108_{group}_{model}_{arm}.json"))["final"] for arm, alpha in ARMS.items()}
    return rule(nat, finals)


def select_vitb(model, out="out/p103"):
    finals = {alpha: json.load(open(f"{out}/p102_{model}_{arm}.json")) for arm, alpha in ARMS.items()}
    nat = next(iter(finals.values()))["native"]
    return rule(nat, {alpha: r["repaired"] for alpha, r in finals.items()})


def available_vitb(out="out/p103"):
    return all(os.path.exists(f"{out}/p102_{m}_{a}.json") for m in VITB for a in ARMS)


def available(group, out="out/p108"):
    return all(os.path.exists(f"{out}/p108_{group}_{m}_{a}.json") for m in MODELS for a in ("native", *ARMS))


def main(argv):
    check = "--check" in argv
    groups = [a for a in argv if a != "--check"] or list(SELECTED)
    missing = [g for g in groups if not available(g)]
    if missing:
        sys.exit(f"p108 outputs missing for {missing} (out/p108); run src/p108.py for native, SD3-12H and SD10-12H first")
    table = {g: {m: select(g, m) for m in MODELS} for g in groups}
    bad = [(g, m, table[g][m], SELECTED[g][m]) for g in groups for m in MODELS if table[g][m] != SELECTED[g][m]]
    if available_vitb():
        table["dogs_vitb"] = {m: select_vitb(m) for m in VITB}
        bad += [("dogs_vitb", m, table["dogs_vitb"][m], SELECTED_VITB[m]) for m in VITB if table["dogs_vitb"][m] != SELECTED_VITB[m]]
    print(json.dumps(table))
    if check:
        for b in bad:
            print("MISMATCH group %s model %s: derived %s, recorded %s" % b)
        print("selection table:", "MISMATCH" if bad else "matches the recorded table")
        sys.exit(1 if bad else 0)
    return table


if __name__ == "__main__":
    main(sys.argv[1:])
