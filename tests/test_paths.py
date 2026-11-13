"""Cache verification (src/paths.py): a cache is used only if it matches its manifest."""
import json

import numpy as np
import pytest

import paths


def test_verify_accepts_matching_and_rejects_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", str(tmp_path))
    monkeypatch.setattr(paths, "_verified", set())
    x, y = np.zeros((4, 3, 2, 2), np.uint8), np.arange(4)
    np.savez(tmp_path / "id_fit.npz", x=x, y=y)
    json.dump(dict(sha1=paths.digest(x, y)), open(tmp_path / "id_fit.json", "w"))
    paths.verify("id_fit", x, y)
    monkeypatch.setattr(paths, "_verified", set())
    x2 = x.copy()
    x2[0, 0, 0, 0] = 1
    with pytest.raises(RuntimeError):
        paths.verify("id_fit", x2, y)


def test_missing_manifest_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", str(tmp_path))
    monkeypatch.setattr(paths, "_verified", set())
    with pytest.raises(RuntimeError):
        paths.verify("id_hold", np.zeros((1, 3, 2, 2), np.uint8), np.zeros(1))


def test_paper_references_present():
    ref = json.load(open(f"{paths.LISTS}/PAPER_CACHES.json"))
    assert {"id_fit", "id_hold", "id_eval"} <= set(ref)
    fresh = json.load(open(f"{paths.LISTS}/FRESH_MANIFEST.json"))
    assert fresh["n"] == len(fresh["images"]) == 2975
