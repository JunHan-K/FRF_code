"""Integrity check of the image splits: no image may occur in two splits, exactly or nearly.
Exact: identical decoded arrays (SHA-1 of the uint8 224x224 image). Near: cosine similarity > 0.98 between mean-centred
16x16 grayscale thumbnails (catches re-encoded or slightly rescaled copies of the same photo). Exits with status 1 if any
pair is found.  usage: python src/check_splits.py"""
import hashlib
import os
import sys

import numpy as np
import torch

from paths import CACHE
SPLITS = ("id_fit", "id_hold", "id_eval", "fresh")
NEAR = 0.98


def features(name):
    x = np.load(os.path.join(CACHE, f"{name}.npz"))["x"]
    hashes, feats = [], []
    for s in range(0, len(x), 250):
        a = np.asarray(x[s:s + 250])
        hashes += [hashlib.sha1(v.tobytes()).hexdigest() for v in a]
        t = torch.from_numpy(a).float().mean(1, keepdim=True)
        f = torch.nn.functional.adaptive_avg_pool2d(t, 16).flatten(1)
        f = f - f.mean(1, keepdim=True)
        feats.append(f / f.norm(dim=1, keepdim=True).clamp_min(1e-6))
    return hashes, torch.cat(feats)


def main():
    H, F = {}, {}
    for n in SPLITS:
        H[n], F[n] = features(n)
    bad = 0
    for i, a in enumerate(SPLITS):
        within = len(H[a]) - len(set(H[a]))
        bad += within
        print(f"{a}: {len(H[a])} images, exact duplicates within {within}")
        for b in SPLITS[i + 1:]:
            exact = len(set(H[a]) & set(H[b]))
            sim = F[a] @ F[b].T
            near = int((sim > NEAR).sum())
            bad += exact + near
            print(f"{a} vs {b}: exact {exact}, near {near}, max cosine {float(sim.max()):.4f}")
    print("OK: no duplicates across splits" if bad == 0 else f"FAIL: {bad} duplicate pairs")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
