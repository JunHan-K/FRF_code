# Recovered, Not Restored: Functional Reference Frames for Vision Transformer Blocks

Code for the paper *Recovered, Not Restored: Functional Reference Frames for Vision Transformer Blocks*.

Functional Reference Frames (FRF) measure what a ViT block does relative to the other blocks of the same network,
in the Fisher metric of the network output. The paper uses FRF to show that recovery after block damage does not
restore the damaged blocks, and applies it to block removal and to class unlearning.

## Layout

All scripts are in `src/` (flat) and are run from the repository root. Outputs go to `out/`, data to `data/`.
Experiment scripts are named `pNN.py` after the analyses of the paper.

| file | role |
|---|---|
| `src/build_cache.py`, `src/build_fresh.py`, `src/check_splits.py` | image splits and their integrity check |
| `src/common.py`, `src/imagenet_subsets.py` | data paths, class index, ImageNet-R / -A class lists |
| `src/surgery.py`, `src/deit_perm.py` | model loading, block permutation, recovery by distillation |
| `src/s12r.py`, `src/e13.py`, `src/p47train.py`, `src/frfx.py` | model registry; training of the disrupted-and-recovered networks |
| `src/p57.py`, `src/spec.py`, `src/p110.py` | FRF measurement (Fisher Gram of block effects, out-of-frame share) |
| `src/p10.py`-`src/p93.py` | recovery analyses (trajectories, rewinding, replacement, mixtures, controls) |
| `src/p18.py` | block removal (selection, recovery, accuracy, throughput) and shared training / evaluation helpers |
| `src/p96.py`-`src/p120.py`, `src/unl118.py` | class unlearning: block damage, group-free LoRA repair, baselines, attacks, rewinding |
| `src/select_ours.py` | damage strength of the unlearning method per group and model, chosen on `id_hold` only |
| `src/spotter_sampling.py` | batch composition of the Spotter baseline |
| `tests/` | CPU tests (no data, no GPU) |

## Data

No images are redistributed. Put under `data/`:

- ImageNet-V2 matched-frequency (`imagenetv2-matched-frequency-format-val`) and, under `data/v2new/`, the
  threshold-0.7 and top-images variants;
- ImageNet-A and ImageNet-R as parquet files (`data/hf_a`, `data/hf_r`);
- `imagenet_class_index.json`.

```
python src/build_cache.py     # id_fit 8,000 / id_eval 24 / id_hold 1,976 (matched-frequency, seed 0); ImageNet-A; ImageNet-R (5,000)
python src/build_fresh.py     # fresh 2,975 (other two V2 variants, sha256-deduplicated against matched-frequency, <= 3 per class)
python src/check_splits.py    # fails if any image occurs in two splits, exactly or as a near duplicate
```

Images are resized to 256 (shorter side) and centre-cropped to 224. Every cache file is written with a manifest of its
sources and a hash; an existing cache that does not match is not reused.

Use of the splits: labels of `id_hold` and `fresh` are used only to measure accuracy. Recovery, repair, block
selection for removal, coefficient fitting and FRF calibration use `id_fit`. Unlearning configurations are selected on
`id_hold`; their final numbers are measured on `fresh`. One exploratory analysis (`src/p82.py`) uses `fresh[:1000]` as a
stopping check and evaluates on the disjoint `fresh[1000:]`. The noise control matches its step-0 damage to the
relocation by output KL on `id_hold` images (no labels). ImageNet-R / -A are evaluated inside their official 200 classes.

## Models

ImageNet-1k timm checkpoints: DeiT-S/B, DeiT III-S/B, AugReg ViT-S/B, and further ViTs for the architecture
checks (`src/s12r.py`, `src/frfx.py`). `src/surgery.py` fetches `model.safetensors` from the Hugging Face hub
(`timm/<name>`) on first use.

## Running

```
pip install -r requirements.txt
python src/s12r.py deit3_s D_La                       # one disrupted-and-recovered network (blocks 6-11 permuted)
python src/p57.py deit3_s L a 6                       # FRF measurement of its recovered block in slot 6
python src/p18.py prune23 deit_s skipG lora1          # block removal, main-text recipe (P18_RECIPE=cheap: rank-1 recipe)
python src/p120.py birds deit_s OURS                  # unlearning: damage + group-free repair, attacks, FRF, rewinding
```

Each script states its arguments in its docstring (`usage:`). Most runs need one GPU with 10 GB or more.
Environment variables that change step counts are for quick tests only; the defaults are the settings of the paper.

## Checks

```
python src/check_splits.py            # no image in two splits (exact or near duplicate); needs the caches
python src/select_ours.py --check     # re-derives the damage strengths of the unlearning method from out/p108 (and out/p103)
                                      # and fails if they differ from the table that p120 / unl118 use
pip install pytest pyflakes && python -m pytest -q tests && python -m pyflakes src tests   # CPU only, no data
```

`tests/` checks the Spotter batch composition (64 images per step and no image twice, also for groups with fewer than
32 classes, and the unchanged draw for groups with at least 32), the selection rule, and that the unlearning scripts take
their damage strengths from `src/select_ours.py`. The same checks run on every push (`.github/workflows/ci.yml`).
`src/p120.py` also re-derives the selection from `out/p108` before an `OURS` run whenever those files exist, and stops on
a difference.

## Reproducibility

- Versions: `requirements.txt` lists the versions used for the paper (Python 3.10, CUDA 12.1).
- Seeds: every script fixes its seeds (seed 0 unless the script takes a seed argument); data order and augmentation use
  seeded generators.
- GPU kernels: the paper runs used the default (non-deterministic) cuDNN / cuBLAS kernels, so a rerun matches the reported
  numbers up to small run-to-run differences, not bit for bit. `FRF_DETERMINISTIC=1` switches to deterministic kernels where
  PyTorch provides them (slower).
- Data: the splits are rebuilt from the public datasets by `src/build_cache.py` / `src/build_fresh.py`; each cache stores
  a manifest with per-file SHA-256 hashes, and `src/check_splits.py` verifies that the splits are disjoint.

## License

MIT, see `LICENSE`.
