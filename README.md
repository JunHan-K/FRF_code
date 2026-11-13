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
| `src/p57.py`, `src/frame.py`, `src/spec.py`, `src/p110.py` | FRF measurement (Fisher Gram of block effects, out-of-frame share) |
| `src/paths.py`, `src/simplex.py`, `src/check_nu.py` | data paths and cache verification; simplex least squares; numerical check of the share |
| `src/p10.py`-`src/p93.py` | recovery analyses (trajectories, rewinding, replacement, mixtures, controls) |
| `src/p18.py` | block removal (selection, recovery, accuracy, throughput) and shared training / evaluation helpers |
| `src/p96.py`-`src/p120.py`, `src/unl118.py` | class unlearning: block damage, group-free LoRA repair, baselines, attacks, rewinding |
| `src/select_ours.py` | damage strength of the unlearning method per group and model, chosen on `id_hold` only |
| `src/spotter_sampling.py` | batch composition of the Spotter baseline |
| `tests/`, `data_lists/` | CPU tests (no data, no GPU); fresh-set image list and paper cache hashes |

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

`FRF_DATA` (default `data`) and `FRF_CACHE` (default `data/cache`) move these folders for every script.
Images are resized to 256 (shorter side) and centre-cropped to 224. Every cache file is written with a manifest of its
sources and a hash; an existing cache that does not match is not reused.

Use of the splits: labels of `id_hold` and `fresh` are used only to measure accuracy. Recovery, repair, block
selection for removal, coefficient fitting and FRF calibration use `id_fit`. Unlearning configurations are selected on
`id_hold`; their final numbers are measured on `fresh`. One exploratory analysis (`src/p82.py`) uses `fresh[:1000]` as a
stopping check and evaluates on the disjoint `fresh[1000:]`. The noise control matches its step-0 damage to the
relocation by output KL on `id_hold` images (no labels). ImageNet-R / -A are evaluated inside their official 200 classes.

## Pipelines and intermediate outputs

Every script reads the outputs of earlier scripts from fixed folders under `out/`. Inputs that come from another script:

| folder / file | written by | read by |
|---|---|---|
| `out/s12r_<model>[_seed<n>]/D_<L|C><p>.pt` | `src/s12r.py <model> D_L<p>` (seed: `DS_SEED=<n>`) | `e13.paths`, p10-p62, p66-p76 |
| `out/s12r_<model>[_seed<n>]/D_<E|N|U><p>.pt` | `src/p47train.py <model> <p> [noise\|unperturbed]` (N needs the D_L runs) | `p47.e_paths`, p47-p62 |
| `out/s12x_<model>_seed<n>/D_<kind><p>.pt` | `src/frfx.py train <model> <kind> <p>` | `frfx.py measure / replace` (`<p>` or `<p>_seed<n>`) |
| `out/p51/<model>/D_<run>{.pt,_ck<t>.pt,.json,_noise.json}` | `P47_OUTDIR=out/p51/<model> DS_SAVE_CKPT=1 python src/p47train.py <model> <p> [late\|noise]` | p61, p79, p80 |
| `out/p57/p57_<model>_<kind>_<p>_<slot>.npz` | `src/p57.py` (`P57_FRESH=1` adds the fresh set) | `src/check_nu.py`, p66, p68, p72, p76 |
| `out/p57ck/..._p51ck<t>.npz` | `P57_PT=out/p51/<model>/D_<run>_ck<t>.pt P57_OUT=out/p57ck P57_TAG=_p51ck<t> python src/p57.py ...` | p80 |
| `out/p65`, `out/p66par` / `out/p65rep`, `out/p66rep` | `P57_OUT=... src/p57.py` / `P59_OUT=... src/p59.py` on the further parents (perms g-l) | p66, p68, p69 |
| `out/p25/<model>_<p>_<slot>_{kdref,lens}.json` | `P14_TAG={kdref,lens} python src/p14.py <model> <p> <slot>` | p37ex, p43 |
| `out/p18/p23_<model>_<crit>_lora1_fix_{main,cheap}.json` | `src/p18.py prune23 <model> <crit> lora1` (`P18_RECIPE=main` / `cheap`) | p55, p60, p88, p90, p96, p116 |
| `out/p79/...` | `src/p79.py` | p81, p82 |
| `out/p104w/<model>_<arm>.pt` | `src/p104.py` (fp16 weights) | p106, p110-p115 |
| `out/p108/`, `out/p103/` | `src/p108.py`, `P102_OUT=out/p103 python src/p102.py` | `src/select_ours.py`, p120 |

A script whose input is missing stops with an error that names the producing command (for p37ex / p43 / p47train), or
with a file-not-found error for the path in the table.

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
python src/check_nu.py out/p57        # out-of-frame share: preregistered cut-off vs other solvers on the stored Grams
pip install pytest pyflakes && python -m pytest -q tests && python -m pyflakes src tests   # CPU only, no data
```

`tests/` checks the out-of-frame share on known answers (inside / outside the reference span, rank-deficient
references), the simplex solver (including degenerate input), cache verification, the Spotter batch composition (64 images per step and no image twice, also for groups with fewer than
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
  a manifest with per-file SHA-256 hashes, and `src/check_splits.py` verifies that the splits are disjoint. Every training
  and evaluation script checks the cache it loads against its manifest. `data_lists/` holds the fresh-set image list
  (paths, SHA-256, labels) and the array hashes of the id splits used for the paper; the build scripts report whether a
  rebuilt split is identical to the paper's.
- Out-of-frame share: `src/frame.py` (eigenvalue cut-off 1e-8 relative to the largest eigenvalue, the preregistered
  value). The older ridge estimate in `src/spec.py` (`nu_span`) is used only by the earlier analyses p34-p56.

## License

MIT, see `LICENSE`.
