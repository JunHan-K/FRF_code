"""Depth branch Phase 1 (depth/PREREG.md): depth surgery CaiT-XXS36 -> 24 blocks, distillation recovery, and the
functional position of every surviving block (argmax linear CKA against the native pretrained XXS24) over training.
usage: surgery.py <run>   run in V_uni V_first V_last V_mid C1 C2"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import huggingface_hub as hf
from safetensors.torch import load_file
from torchvision.ops import roi_align

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

# FRF_DETERMINISTIC=1: deterministic GPU kernels where PyTorch has them (slower; not used for the paper runs, see README)
if os.environ.get("FRF_DETERMINISTIC", "0").strip().lower() in ("1", "true", "yes", "on"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

CACHE = paths.CACHE
OUT = "out/depth"
M36, M24 = "cait_xxs36_224.fb_dist_in1k", "cait_xxs24_224.fb_dist_in1k"
KEEP = {"V_uni": [i for i in range(36) if i % 3 != 2], "V_first": list(range(24)), "V_last": list(range(12, 36)),
        "V_mid": list(range(12)) + list(range(24, 36)),
        # supplementary 36 -> 18 cuts (functional target: relative position k * 24 / 18 in native XXS24 coordinates)
        "H_uni": list(range(0, 36, 2)), "H_first": list(range(18)), "H_last": list(range(18, 36))}
# supplement 4 (post-hoc descriptive): shuffled order, random subsets
KEEP["V_perm"] = [KEEP["V_uni"][i] for i in np.random.default_rng(0).permutation(24)]
# supplement 5: cut the native XXS24 to 16 blocks; positions read against XXS36 (target k * 36 / 16)
KEEP.update({"N_first": list(range(16)), "N_last": list(range(8, 24)), "N_uni": [i for i in range(24) if i % 3 != 2]})
KEEP["V_permL"] = [list(range(12, 36))[i] for i in np.random.default_rng(1).permutation(24)]
KEEP["C1"] = list(range(36))
KEEP.update({f"V_permL{s}": [list(range(12, 36))[i] for i in np.random.default_rng(s).permutation(24)] for s in (2, 3)})
KEEP.update({f"V_permF{s}": [list(range(24))[i] for i in np.random.default_rng(s).permutation(24)] for s in (1, 2)})
KEEP["V_mid2"] = list(range(6)) + list(range(18, 36))
KEEP.update({f"V_rand{s}": sorted(np.random.default_rng(s).choice(36, 24, replace=False).tolist()) for s in (1, 2, 3, 4)})
STEPS, BS, LR, WD, WARM = 3000, 64, float(os.environ.get("DS_LR", 1e-4)), 0.05, 100
MB = int(os.environ.get("DS_MB", 16))
CKPT = [0, 100, 300, 1000, 3000, 6000, 8000, 10000, 30000]  # 6000+ only reached by long runs (10000/30000: PREREG86)
SMOKE = int(os.environ.get("DS_SMOKE", 0))
SEED = int(os.environ.get("DS_SEED", 0))  # supplementary seeds: data order and augmentation only
LOSS = os.environ.get("DS_LOSS", "kd")  # supplementary: "ce" = label cross-entropy, no teacher
TRAIN = os.environ.get("DS_TRAIN", "all")  # supplementary: "ln" = LayerNorm params and biases; "attn" / "mlp" = only those
dev = "cuda"


def load(n):
    m = timm.create_model(n, pretrained=False)
    m.load_state_dict(load_file(hf.hf_hub_download("timm/" + n, "model.safetensors")))
    return m.to(dev)


def cache(name):
    d = np.load(os.path.join(CACHE, f"{name}.npz"))
    x, y = d["x"], d["y"]
    paths.verify(name, x, y)
    if SMOKE:
        x, y = x[:SMOKE], y[:SMOKE]
    return torch.from_numpy(x), torch.from_numpy(y)


class Norm(nn.Module):
    def __init__(self, m):
        super().__init__()
        c = m.pretrained_cfg
        self.register_buffer("mu", torch.tensor(c["mean"]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor(c["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mu) / self.sd


def augment(x, g):
    """x uint8 [B,3,224,224] on GPU -> float in [0,1]; RandomResizedCrop(224, scale 0.35-1, ratio 3/4-4/3) + flip."""
    B = len(x)
    x = x.float().div_(255)
    area = torch.empty(B).uniform_(0.35, 1.0, generator=g) * 224 * 224
    logr = torch.empty(B).uniform_(math.log(3 / 4), math.log(4 / 3), generator=g)
    w = (area * logr.exp()).sqrt().clamp(16, 224)
    h = (area / logr.exp()).sqrt().clamp(16, 224)
    x0 = torch.rand(B, generator=g) * (224 - w)
    y0 = torch.rand(B, generator=g) * (224 - h)
    boxes = torch.stack([torch.arange(B).float(), x0, y0, x0 + w, y0 + h], 1).to(x.device)
    out = roi_align(x, boxes, (224, 224), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flip = (torch.rand(B, generator=g) < 0.5).to(x.device)
    out[flip] = out[flip].flip(-1)
    return out


@torch.no_grad()
def block_feats(m, nrm, xs, bs=128):
    st = {}
    hs = [b.register_forward_hook(lambda mod, i, o, k=k: st.__setitem__(k, o)) for k, b in enumerate(m.blocks)]
    pm, lg = [[] for _ in m.blocks], []
    for s in range(0, len(xs), bs):
        lg.append(m(nrm(xs[s:s + bs].to(dev).float().div_(255))).float())
        for k in range(len(m.blocks)):
            pm[k].append(st[k].mean(1).double())
    for h in hs:
        h.remove()
    return [torch.cat(p) for p in pm], torch.cat(lg)


def gram_c(X):
    X = X - X.mean(0)
    return X, (X.T @ X).norm()


def cka_matrix(A, B):
    Ac, Bc = [gram_c(a) for a in A], [gram_c(b) for b in B]
    return np.array([[float((a.T @ b).norm() ** 2 / (na * nb)) for b, nb in Bc] for a, na in Ac])


def main(run):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    teacher = load(M36).eval()
    native = load(M24).eval()
    nrm = Norm(teacher).to(dev)
    xh, yh = cache("id_hold")
    xf, yf = cache("id_fit")
    xf = xf.pin_memory()  # on GPU it pushed the 10 GB MIG slice over the limit
    nat_f, _ = block_feats(native, nrm, xh)
    tea_f, lp_t_hold = block_feats(teacher, nrm, xh)
    lp_t_hold = lp_t_hold.log_softmax(-1)
    if run.startswith("N_"):
        student = load(M24)
        src = KEEP[run]
        student.blocks = nn.Sequential(*[student.blocks[i] for i in src])
        nat_f = tea_f  # reference: XXS36
    elif run == "C2":
        student = load(M24)
        src = list(range(24))
    elif run == "C3_scratch":  # supplement 4: random-init XXS24, no pretrained weights
        student = timm.create_model(M24, pretrained=False).to(dev)
        src = list(range(24))
    else:
        student = load(M36)
        src = KEEP.get(run, list(range(36)))
        student.blocks = nn.Sequential(*[student.blocks[i] for i in src])
    student.train()
    params = list(student.parameters())
    if TRAIN == "ln":
        params = [p for n, p in student.named_parameters() if n.endswith(".bias") or "norm" in n]
        for p in student.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
    elif TRAIN in ("attn", "mlp"):
        params = [p for n, p in student.named_parameters() if n.startswith("blocks.") and f".{TRAIN}." in n]
        for p in student.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR * (10 if TRAIN == "ln" else 1), weight_decay=WD if TRAIN != "ln" else 0.0)
    steps = int(os.environ.get("DS_STEPS", STEPS))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(SEED)
    ckpts = sorted({c for c in CKPT if c <= steps} | {steps})
    rec = dict(run=run, src=src, ckpt=[], cka_native_t0=None)

    def measure(t):
        student.eval()
        f, lg = block_feats(student, nrm, xh)
        C = cka_matrix(f, nat_f)                       # [L_student, 24]
        Cs = cka_matrix(f, [tea_f[i] for i in src]) if run not in ("C2", "C3_scratch") and not run.startswith("N_") else None  # vs own source block
        lp = lg.log_softmax(-1)
        r = dict(step=t, pos=C.argmax(1).tolist(), cka_native=C.tolist(),
                 cka_source_diag=(np.diag(Cs).tolist() if Cs is not None else None),
                 top1=float((lg.argmax(1).cpu() == yh).float().mean()),
                 kl_teacher=float((lp_t_hold.exp() * (lp_t_hold - lp)).sum(-1).mean()))
        rec["ckpt"].append(r)
        print(run, "step", t, "top1 %.4f kl %.4f" % (r["top1"], r["kl_teacher"]), "pos", r["pos"], flush=True)
        student.train()

    measure(0)
    for t in range(1, steps + 1):
        idx = torch.randint(len(xf), (BS,), generator=g)
        x = nrm(augment(xf[idx].to(dev, non_blocking=True), g))
        opt.zero_grad(set_to_none=True)
        for s in range(0, BS, MB):  # micro-batches, same effective batch
            lps = student(x[s:s + MB]).log_softmax(-1)
            if LOSS == "ce":
                loss = F.nll_loss(lps, yf[idx[s:s + MB]].to(dev).long(), reduction="sum") / BS
            else:
                with torch.no_grad():
                    lpt = teacher(x[s:s + MB]).log_softmax(-1)
                loss = (lpt.exp() * (lpt - lps)).sum(-1).sum() / BS
            loss.backward()
        opt.step()
        sched.step()
        if t in ckpts:
            measure(t)
    tag = (f"{run}{f'_s{SEED}' if SEED else ''}{f'_{TRAIN}' if TRAIN != 'all' else ''}{f'_steps{steps}' if steps != STEPS else ''}{f'_lr{LR:g}' if LR != 1e-4 else ''}{'_ce' if LOSS == 'ce' else ''}"
           f"{'_smoke' if SMOKE else ''}")
    json.dump(rec, open(os.path.join(OUT, f"{tag}.json"), "w"))
    if not SMOKE:
        torch.save(student.state_dict(), os.path.join(OUT, f"{tag}.pt"))


if __name__ == "__main__":
    main(sys.argv[1])
