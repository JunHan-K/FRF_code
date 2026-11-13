"""Shared helpers: model registry, checkpoint paths of the disrupted-and-recovered networks, token embedding."""
import os
import sys


sys.path.insert(0, "src")
import s12r  # noqa: E402  (registers the permutations in deit_perm.KEEP)

MODELS = s12r.MODELS


SEED_SFX = os.environ.get("E13_SEED_SFX", "")  # e.g. "_seed1": seed-1 replicate checkpoints


def paths(model, tag):
    if SEED_SFX:
        base = f"out/s12r_{model}{SEED_SFX}"
        return f"{base}/{tag}.pt", f"{base}/dswap_{tag}.json", f"{base}/{tag}.json"
    base = f"out/s12r_{model}"  # written by src/s12r.py <model> <tag>
    return f"{base}/{tag}.pt", f"{base}/dswap_{tag}.json", f"{base}/{tag}.json"


def emb(m, x):
    return m.norm_pre(m._pos_embed(m.patch_embed(x)))
