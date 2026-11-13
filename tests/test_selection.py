"""Damage-strength selection (src/select_ours.py): the rule, and that p120 / unl118 take their values from it."""
import ast
import json
import os

import select_ours as SEL

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def rec(fh, rh):
    return {"forget_hold": fh, "retain_hold": rh}


def test_rule_smallest_drop_among_passing():
    nat = rec(0.80, 0.70)
    assert SEL.rule(nat, {3: rec(0.00, 0.67), 10: rec(0.05, 0.69)}) == 10   # both pass, 10 has the smaller drop
    assert SEL.rule(nat, {3: rec(0.00, 0.67), 10: rec(0.09, 0.69)}) == 3    # 10 forgets too little (> 0.1 x native)
    assert SEL.rule(nat, {3: rec(0.00, 0.64), 10: rec(0.00, 0.60)}) is None  # both lose > 5 points


def test_select_reads_p108_records(tmp_path):
    for m in SEL.MODELS:
        json.dump({"native": rec(0.8, 0.7)}, open(tmp_path / f"p108_g_{m}_native.json", "w"))
        json.dump({"final": rec(0.0, 0.66)}, open(tmp_path / f"p108_g_{m}_SD3-12H.json", "w"))
        json.dump({"final": rec(0.0, 0.69)}, open(tmp_path / f"p108_g_{m}_SD10-12H.json", "w"))
    assert SEL.available("g", out=str(tmp_path))
    assert {m: SEL.select("g", m, out=str(tmp_path)) for m in SEL.MODELS} == {m: 10 for m in SEL.MODELS}


def test_table_has_every_group_and_model():
    assert set(SEL.SELECTED) == {"dogs", "birds", "reptiles", "insects"}
    for g in SEL.SELECTED.values():
        assert set(g) == set(SEL.MODELS) and set(g.values()) <= set(SEL.ARMS.values())
    assert set(SEL.SELECTED_VITB) == set(SEL.VITB)


def _assigned(path, name):
    tree = ast.parse(open(os.path.join(SRC, path)).read())
    return [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def test_scripts_use_the_table():
    """No hand-copied alpha values: p120 and unl118 build OURS_ALPHA from select_ours."""
    assert _assigned("p120.py", "OURS_ALPHA") == ["{g: SEL.SELECTED[g] for g in GROUPS}"]
    (u,) = _assigned("unl118.py", "OURS_ALPHA")
    assert "SEL.SELECTED['dogs']" in u and "SEL.SELECTED_VITB" in u


def test_availability_is_per_model(tmp_path):
    """a missing model must not disable the check for the models whose runs exist"""
    for a in ("native", "SD3-12H", "SD10-12H"):
        json.dump({}, open(tmp_path / f"p108_g_deit_s_{a}.json", "w"))
    assert SEL.available("g", out=str(tmp_path), models=("deit_s",))
    assert not SEL.available("g", out=str(tmp_path))
