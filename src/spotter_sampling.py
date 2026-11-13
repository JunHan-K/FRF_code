"""Batch composition of the Spotter baseline (p108): 32 forget-class slots with 2 images each, 64 images per step,
no image twice. Kept free of data and model imports so that it can be tested on CPU (tests/test_spotter_sampling.py)."""
import torch


def spotter_classes(n, k, g):
    """k forget classes for one Spotter step (2 images each). With n < k classes (insects: 27) every class is used once and
    the remaining k - n slots are filled with further distinct classes, so a step always has 2k = 64 images."""
    if n >= k:
        return torch.randperm(n, generator=g)[:k]
    return torch.cat([torch.randperm(n, generator=g), torch.randperm(n, generator=g)[:k - n]])  # k - n <= n: no class more than twice


def spotter_images(byc, cls, g):
    """2 images per class slot; a class drawn twice gets 4 distinct images (one permutation per class, consumed in pairs),
    so no image occurs twice in a step."""
    perm, pos, out = {}, {}, []
    for c in cls.tolist():
        if c not in perm:
            perm[c], pos[c] = torch.randperm(len(byc[c]), generator=g), 0
        pair = perm[c][pos[c]:pos[c] + 2]
        if len(pair) < 2:
            raise ValueError(f"class slot {c} drawn twice needs >= 4 images, has {len(byc[c])}")
        out.append(byc[c][pair])
        pos[c] += 2
    return torch.stack(out).flatten()
