"""Spotter batch composition (src/spotter_sampling.py): every step has 64 images and no image twice."""
import pytest
import torch

from spotter_sampling import spotter_classes, spotter_images

K = 32  # class slots per step (batch 64, 2 images each)


def groups(n, per=8):
    """n classes with `per` images each; image ids are global and distinct."""
    return [torch.arange(c * per, (c + 1) * per) for c in range(n)]


@pytest.mark.parametrize("n", [17, 27, 31, 32, 36, 59, 118])
def test_64_distinct_images_every_step(n):
    byc = groups(n)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        cls = spotter_classes(n, K, g)
        img = spotter_images(byc, cls, g)
        assert img.numel() == 2 * K
        assert img.unique().numel() == 2 * K
        counts = torch.bincount(cls, minlength=n)
        assert counts.max() <= 2
        if n < K:
            assert (counts >= 1).all()          # every class used
            assert int((counts == 2).sum()) == K - n
        else:
            assert counts.max() == 1             # distinct classes


def test_unchanged_draw_with_enough_classes():
    """With >= 32 classes the draw is the original one: randperm(n)[:32], then randperm(len(class))[:2] per class."""
    n, byc = 59, groups(59)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    cls = spotter_classes(n, K, g1)
    img = spotter_images(byc, cls, g1)
    ref_cls = torch.randperm(n, generator=g2)[:K]
    ref = torch.stack([byc[c][torch.randperm(len(byc[c]), generator=g2)[:2]] for c in ref_cls.tolist()]).flatten()
    assert torch.equal(cls, ref_cls) and torch.equal(img, ref)


def test_class_drawn_twice_needs_four_images():
    byc = [torch.arange(c * 3, c * 3 + 3) for c in range(27)]  # only 3 images per class
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        spotter_images(byc, spotter_classes(27, K, g), g)
