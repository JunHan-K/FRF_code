"""Fresh evaluation set: ImageNet-V2 threshold-0.7 and top-images, minus every image whose file
content (sha256) also appears in matched-frequency (the images behind id_fit / id_hold / id_eval). Class-balanced: at most
3 images per class, drawn with numpy RandomState(0) from the sorted list of unique (by hash) remaining images.
Same preprocessing as build_cache.py (resize 256, centre crop 224, uint8). Writes data/cache/fresh.npz and
data/v2new/FRESH_MANIFEST.json (paths, hashes, seed, preprocessing). Run src/check_splits.py afterwards.
usage: python src/build_fresh.py"""
import glob
import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

from paths import CACHE, DATA, FRESH_MANIFEST, LISTS


def load_path(path):  # identical to build_cache.prep (resize shorter side 256 bilinear, centre crop 224); inlined to avoid pyarrow
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8).transpose(2, 0, 1)


PER_CLASS, SEED = 3, 0


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    mf = sorted(glob.glob(os.path.join(DATA, "imagenetv2-matched-frequency-format-val", "*", "*")))
    new = sorted(glob.glob(os.path.join(DATA, "v2new", "imagenetv2-threshold0.7-format-val", "*", "*")) +
                 glob.glob(os.path.join(DATA, "v2new", "imagenetv2-top-images-format-val", "*", "*")))
    with Pool(int(os.environ.get("NPROC", 12))) as pool:
        h_mf = set(pool.map(sha, mf, chunksize=64))
        h_new = pool.map(sha, new, chunksize=64)
    seen, cand = set(), {}
    for p, h in zip(new, h_new):
        if h in h_mf or h in seen:
            continue
        seen.add(h)
        cand.setdefault(int(p.split(os.sep)[-2]), []).append((p, h))
    rng = np.random.RandomState(SEED)
    pick = []
    for c in sorted(cand):
        L = cand[c]
        for i in sorted(rng.choice(len(L), min(PER_CLASS, len(L)), replace=False)):
            pick.append((L[i][0], L[i][1], c))
    images = [dict(path=os.path.relpath(p, DATA), sha256=h, label=c) for p, h, c in pick]
    paper = json.load(open(os.path.join(LISTS, "FRESH_MANIFEST.json")))
    print("fresh selection:", "identical to" if paper["images"] == images else "DIFFERS from", "the paper's (data_lists/FRESH_MANIFEST.json)",
          flush=True)
    npz, man_path = os.path.join(CACHE, "fresh.npz"), FRESH_MANIFEST
    if os.path.exists(npz) and os.environ.get("FRF_REBUILD") != "1":
        # reuse an existing fresh set only if it is exactly the set this script selects and the file is unchanged
        old = json.load(open(man_path)) if os.path.exists(man_path) else {}
        if old.get("images") == images and old.get("npz_sha256") == sha(npz):
            print("fresh.npz verified against FRESH_MANIFEST.json; not rebuilt")
            return
        raise RuntimeError(f"{npz} does not match the selected images or its manifest; delete it or set FRF_REBUILD=1")
    with Pool(int(os.environ.get("NPROC", 12))) as pool:
        x = np.stack(pool.map(load_path, [p for p, _, _ in pick], chunksize=32))
    y = np.asarray([c for _, _, c in pick], dtype=np.int64)
    np.savez(npz, x=x, y=y)
    man = dict(n=len(pick), per_class=PER_CLASS, seed=SEED, n_mf=len(mf), n_new_files=len(new), n_unique_not_in_mf=len(seen),
               n_classes=len(cand), preprocessing="resize shorter side 256 (bilinear), centre crop 224, uint8 CHW",
               npz_sha256=sha(npz), images=images)
    json.dump(man, open(man_path, "w"), indent=0)
    print("fresh.npz:", "identical to" if man["npz_sha256"] == paper["npz_sha256"] else "differs from (image decoding / library versions)",
          "the file used for the paper", flush=True)
    print({k: v for k, v in man.items() if k != "images"})


if __name__ == "__main__":
    main()
