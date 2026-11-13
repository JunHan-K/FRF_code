"""Model registry and block permutations; trains one disrupted-and-recovered network (deit_perm.main).
Runs: D_C (unperturbed re-training), D_L<s> (blocks 6-11 permuted by NEW_PI / deit_perm.PI), D_E<s> (blocks 0-5 permuted).
usage: s12r.py <model> <run>      (S_SEED / surgery.SEED selects the training seed)"""
import os
import sys


sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402

MODELS = {"deit3_s": "deit3_small_patch16_224.fb_in1k", "augreg_s": "vit_small_patch16_224.augreg_in21k_ft_in1k",
          "deit_s": "deit_small_patch16_224.fb_in1k",
          # PREREG17 ViT-B scale-up
          "deit_b": "deit_base_patch16_224.fb_in1k", "deit3_b": "deit3_base_patch16_224.fb_in1k",
          "augreg_b": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
          # PREREG24 breadth (ViT family)
          "vit_t": "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit_l": "vit_large_patch16_224.augreg_in21k_ft_in1k",
          "deit_bd": "deit_base_distilled_patch16_224.fb_in1k", "clip_b": "vit_base_patch16_clip_224.openai_ft_in1k"}
NEW_PI = {"c": [3, 1, 4, 5, 2, 0], "d": [3, 2, 4, 0, 5, 1], "e": [4, 1, 0, 5, 3, 2], "f": [0, 1, 4, 5, 2, 3]}  # PREREG3 E0
NEW_PI.update({"g": [2, 4, 1, 5, 3, 0], "h": [5, 1, 0, 3, 2, 4], "i": [1, 4, 0, 3, 2, 5]})  # PREREG65: default_rng(65), >= 4 moved, not a..f
NEW_PI.update({"j": [1, 0, 5, 4, 3, 2], "k": [1, 2, 5, 0, 3, 4], "l": [2, 3, 1, 0, 5, 4]})  # PREREG66: default_rng(66), >= 4 moved, not a..i
NEW_PI.update({"m": [4, 2, 3, 5, 1, 0], "n": [4, 0, 2, 1, 5, 3], "o": [2, 3, 5, 4, 1, 0], "p": [1, 5, 0, 2, 4, 3], "q": [3, 1, 5, 2, 4, 0],
               "r": [4, 5, 3, 2, 0, 1]})  # PREREG67: default_rng(67), >= 4 moved, not a..l
for _s, _p in NEW_PI.items():
    D.KEEP[f"D_L{_s}"] = list(range(6)) + [6 + i for i in _p]
for _s in "cdghijklmnopqr":  # PREREG58 (c, d), PREREG65 (g-i), PREREG66 (j-l), PREREG67 (m-r): early-shuffled replicates use the same permutations on blocks 0..5
    D.KEEP[f"D_E{_s}"] = NEW_PI[_s] + list(range(6, 12))


def main(model, run):
    D.M = MODELS[model]
    sfx = f"_seed{S.SEED}" if S.SEED else ""  # training-seed replicates
    S.OUT = f"out/s12r_{model}{sfx}"
    os.makedirs(S.OUT, exist_ok=True)
    if not os.path.exists(f"{S.OUT}/{run}.pt"):
        D.main(run)


if __name__ == "__main__":
    main(*sys.argv[1:])
