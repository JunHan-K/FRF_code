"""PREREG105 (GPU): PREREG102 on a different forget set - 100 random non-dog ImageNet classes (torch.randperm, seed 105).
p102.py is run unchanged except that the forget set is replaced before import; outputs in out/p105.
Written 2026-11-08 05:59 KST, before any PREREG105 network exists.
usage: p105.py <model> <arm>"""
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src")
import torch  # noqa: E402
import p96  # noqa: E402

_nd = torch.tensor([c for c in range(1000) if not 151 <= c <= 268])
p96.FORGET = _nd[torch.randperm(len(_nd), generator=torch.Generator().manual_seed(105))[:100]].sort().values
os.environ.setdefault("P102_OUT", "out/p105")
import p102  # noqa: E402

assert torch.equal(p102.FORGET, p96.FORGET)

if __name__ == "__main__":
    p102.main(*sys.argv[1:3])
