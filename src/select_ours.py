"""Damage strength of OURS per forget group and model, chosen on id_hold only (PREREG102 rule, applied in PREREG108):
among the damage arms SD3-12H and SD10-12H (selective dampening, alpha 3 / 10, all 12 blocks, head re-init + repair),
keep those with id_hold forget top-1 <= 0.1 x native and id_hold retained top-1 drop <= 5 points, and take the one with
the smallest id_hold retained drop. No fresh-set value enters the choice. Reads the p108 outputs (out/p108).
p120.OURS_ALPHA is this table.
usage: python src/select_ours.py [group ...]      (default: dogs birds reptiles insects)"""
import json
import sys

MODELS = ("deit_s", "deit3_s", "augreg_s")
ARMS = {"SD3-12H": 3, "SD10-12H": 10}


def select(group, model, out="out/p108"):
    nat = json.load(open(f"{out}/p108_{group}_{model}_native.json"))["native"]
    ok = {}
    for arm, alpha in ARMS.items():
        f = json.load(open(f"{out}/p108_{group}_{model}_{arm}.json"))["final"]
        drop = nat["retain_hold"] - f["retain_hold"]
        if f["forget_hold"] <= 0.1 * nat["forget_hold"] and drop <= 0.05:
            ok[alpha] = drop
    return min(ok, key=ok.get) if ok else None


def main(groups):
    table = {g: {m: select(g, m) for m in MODELS} for g in groups}
    print(json.dumps(table))
    return table


if __name__ == "__main__":
    main(sys.argv[1:] or ["dogs", "birds", "reptiles", "insects"])
