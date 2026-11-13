"""Decode images once into uint8 arrays (resize 256, center crop 224).
Each cache file <name>.npz is written with a manifest <name>.json listing its sources (relative paths with the SHA-256
of each source file, and parquet row indices) and a SHA-1 of the stored arrays. An existing cache is reused only if its manifest matches the sources this
script would use and its arrays match the stored hash; otherwise the script stops (set FRF_REBUILD=1 to rebuild).
usage: python src/build_cache.py"""
import glob
import hashlib
import io
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from common import CACHE, DATA, class_index


def prep(im):
    im = im.convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_path(path):
    return prep(Image.open(path))


def load_bytes(b):
    return prep(Image.open(io.BytesIO(b)))


def _paths(name):
    return os.path.join(CACHE, f"{name}.npz"), os.path.join(CACHE, f"{name}.json")


def _digest(x, y):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.asarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()


def done(name, sources):
    """True if a valid cache for exactly these sources exists; raises on a stale or unverifiable cache"""
    npz, man = _paths(name)
    if not os.path.exists(npz):
        return False
    if os.environ.get("FRF_REBUILD") == "1":
        return False
    ok = os.path.exists(man)
    if ok:
        m = json.load(open(man))
        d = np.load(npz)
        ok = m.get("sources") == sources and m.get("sha1") == _digest(d["x"], d["y"])
    if not ok:
        raise RuntimeError(f"{npz} does not match its sources or manifest; delete it or set FRF_REBUILD=1")
    return True


def save(name, x, labels, sources):
    npz, man = _paths(name)
    np.savez(npz, x=x, y=np.asarray(labels, dtype=np.int64))
    json.dump(dict(name=name, n=len(labels), sources=sources, sha1=_digest(x, labels)), open(man, "w"))
    print(name, x.shape, flush=True)


def read_parquets(files, label_fn):
    blobs, labels = [], []
    for f in files:
        t = pq.read_table(f).to_pydict()
        blobs += [d["bytes"] for d in t["image"]]
        labels += label_fn(t)
    return blobs, labels


def main():
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.RandomState(0)
    wnid2idx = class_index()
    with Pool(int(os.environ.get("NPROC", 12))) as pool:
        v2 = sorted(glob.glob(os.path.join(DATA, "imagenetv2-matched-frequency-format-val", "*", "*")))
        lab = [int(p.split(os.sep)[-2]) for p in v2]
        perm = rng.permutation(len(v2))
        splits = {"id_fit": perm[:8000], "id_eval": perm[8000:8024], "id_hold": perm[8024:]}
        for k, idx in splits.items():
            src = [[os.path.relpath(v2[i], DATA), file_sha256(v2[i])] for i in idx]
            if not done(k, src):
                save(k, np.stack(pool.map(load_path, [v2[i] for i in idx], chunksize=32)), [lab[i] for i in idx], src)

        with open(os.path.join(DATA, "hf_a", "dataset_infos.json")) as f:
            names = next(iter(json.load(f).values()))["features"]["label"]["names"]
        files = sorted(glob.glob(os.path.join(DATA, "hf_a", "*.parquet")))
        blobs, labels = read_parquets(files, lambda t: [wnid2idx[names[i]] for i in t["label"]])
        src = dict(files=[[os.path.relpath(f, DATA), file_sha256(f)] for f in files], rows="all")
        if not done("shift_A", src):
            save("shift_A", np.stack(pool.map(load_bytes, blobs, chunksize=32)), labels, src)

        files = sorted(glob.glob(os.path.join(DATA, "hf_r", "*.parquet")))
        blobs, labels = read_parquets(files, lambda t: [wnid2idx[w] for w in t["wnid"]])
        idx = sorted(int(i) for i in rng.choice(len(blobs), 5000, replace=False))
        src = dict(files=[[os.path.relpath(f, DATA), file_sha256(f)] for f in files], rows=idx)
        if not done("shift_R", src):
            save("shift_R", np.stack(pool.map(load_bytes, [blobs[i] for i in idx], chunksize=32)),
                 [labels[i] for i in idx], src)


if __name__ == "__main__":
    main()
