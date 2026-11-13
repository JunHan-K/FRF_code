"""Data locations shared by every script, and verification of the image caches against their manifests.
FRF_DATA (default data) and FRF_CACHE (default data/cache) select the folders; build, check, training and evaluation
scripts all read them from here, so a cache built under FRF_CACHE is the one every script uses."""
import hashlib
import json
import os

import numpy as np

DATA = os.environ.get("FRF_DATA", "data")
CACHE = os.environ.get("FRF_CACHE", os.path.join(DATA, "cache"))
FRESH_MANIFEST = os.path.join(DATA, "v2new", "FRESH_MANIFEST.json")
LISTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_lists")  # references of the paper
_verified = set()


def digest(x, y):
    """SHA-1 of the stored arrays (images and labels), as written to <name>.json by build_cache.py"""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.asarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compare_to_paper(name, sha1):
    """print whether a cache equals the one used for the paper (data_lists/PAPER_CACHES.json), if it is listed there"""
    ref = json.load(open(os.path.join(LISTS, "PAPER_CACHES.json"))).get(name)
    if ref is not None:
        print(f"{name}: {'identical to' if ref['sha1'] == sha1 else 'DIFFERS from'} the cache used for the paper", flush=True)


def verify(name, x, y):
    """Raise unless cache <name> matches its manifest (once per process and cache). fresh: FRESH_MANIFEST.json (file
    SHA-256, written by build_fresh.py); every other cache: <name>.json next to it (array SHA-1, written by build_cache.py)."""
    if name in _verified:
        return
    npz = os.path.join(CACHE, f"{name}.npz")
    if name == "fresh":
        ok = os.path.exists(FRESH_MANIFEST) and json.load(open(FRESH_MANIFEST)).get("npz_sha256") == file_sha256(npz)
        where = FRESH_MANIFEST
    else:
        man = os.path.join(CACHE, f"{name}.json")
        ok = os.path.exists(man) and json.load(open(man)).get("sha1") == digest(x, y)
        where = man
    if not ok:
        raise RuntimeError(f"{npz} does not match {where}; rebuild it with src/build_cache.py / src/build_fresh.py")
    _verified.add(name)
