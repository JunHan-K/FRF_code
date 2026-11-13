"""Depth Phase 2 (depth/PREREG.md): cross-architecture gate. DeiT-S with its early or late half shuffled by a fixed
permutation, KD recovery from the unshuffled original (training as in surgery.py).
usage: deit_perm.py <D_C|D_Ea|D_Eb|D_La|D_Lb>"""
import json
import math
import os
import sys

import torch
import torch.nn as nn

import surgery as S

M = "deit_small_patch16_224.fb_in1k"
PI = {"a": [2, 5, 4, 1, 3, 0], "b": [2, 3, 0, 5, 4, 1]}  # rng(3) / rng(6) permutation(6); slot t <- source PI[t]
KEEP = {"D_C": list(range(12))}
for s, p in PI.items():
    KEEP[f"D_E{s}"] = p + list(range(6, 12))
    KEEP[f"D_L{s}"] = list(range(6)) + [6 + i for i in p]
STEPS = int(os.environ.get("DS_STEPS", S.STEPS))
TRAIN = os.environ.get("DS_TRAIN", "all")  # Phase 3: ctx | perm | ln | lnperm (late-shuffled slots 6..11)
LATE = range(6, 12)


class _LoRA(nn.Module):
    """PREREG67 Q3: frozen linear + rank-r edit B @ A (B zero-init), merged into the weight before saving"""

    def __init__(self, lin, r):
        super().__init__()
        self.lin = lin
        self.A = nn.Parameter(torch.randn(r, lin.in_features, device=lin.weight.device) / math.sqrt(lin.in_features))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device))

    def forward(self, x):
        return self.lin(x) + (x @ self.A.t()) @ self.B.t()


LORA_SITES = (("attn", "qkv"), ("attn", "proj"), ("mlp", "fc1"), ("mlp", "fc2"))


def lora_merge(m):
    """replace every _LoRA by its linear with W + B @ A (state-dict keys as the original network)"""
    with torch.no_grad():
        for blk in m.blocks:
            for a, b in LORA_SITES:
                mod = getattr(getattr(blk, a), b)
                if isinstance(mod, _LoRA):
                    mod.lin.weight += mod.B @ mod.A
                    setattr(getattr(blk, a), b, mod.lin)


def trainable(m):
    """(params, lr, wd) for the Phase 3 training condition"""
    if TRAIN.startswith("lora"):  # PREREG67 Q3: rank-r LoRA on every block (lr 1e-3, no wd: PREREG32 development recipe)
        for q in m.parameters():
            q.requires_grad_(False)
        ps = []
        for blk in m.blocks:
            for a, b in LORA_SITES:
                mod = _LoRA(getattr(getattr(blk, a), b), int(TRAIN[4:]))
                setattr(getattr(blk, a), b, mod)
                ps += [mod.A, mod.B]
        return ps, (1e-3, 0.0)
    named = list(m.named_parameters())
    late = lambda n: any(n.startswith(f"blocks.{k}.") for k in LATE)
    sel = {"all": lambda n: True, "ctx": lambda n: not late(n), "perm": late,
           "ln": lambda n: "norm" in n or n.endswith(".bias") or n.startswith("head."),
           "lnperm": lambda n: late(n) and (".norm1." in n or ".norm2." in n)}[TRAIN]
    for n, p in named:
        p.requires_grad_(sel(n))
    ps = [p for n, p in named if sel(n)]
    return ps, (S.LR * 10, 0.0) if TRAIN in ("ln", "lnperm") else (S.LR, S.WD)


def build(run):
    m = S.load(M)
    m.blocks = nn.Sequential(*[m.blocks[i] for i in KEEP[run]])
    return m


@torch.no_grad()
def evaluate(m, nrm, x, y, lp_t):
    m.eval()
    lg = torch.cat([m(nrm(x[s:s + 128].to(S.dev).float().div_(255))).float() for s in range(0, len(x), 128)])
    lp = lg.log_softmax(-1)
    m.train()
    return float((lg.argmax(1).cpu() == y).float().mean()), float((lp_t.exp() * (lp_t - lp)).sum(-1).mean())


def main(run):
    os.makedirs(S.OUT, exist_ok=True)
    torch.manual_seed(S.SEED)
    teacher = S.load(M).eval()
    nrm = S.Norm(teacher).to(S.dev)
    xh, yh = S.cache("id_hold")
    xf, _ = S.cache("id_fit")
    xf = xf.pin_memory()
    with torch.no_grad():
        lp_t = torch.cat([teacher(nrm(xh[s:s + 128].to(S.dev).float().div_(255))).float()
                          for s in range(0, len(xh), 128)]).log_softmax(-1)
    student = build(run).train()
    ps, (lr, wd) = trainable(student)
    print(run, TRAIN, "trainable params", sum(p.numel() for p in ps), flush=True)
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, STEPS) / STEPS)))
    g = torch.Generator().manual_seed(S.SEED)
    ckpts = sorted({c for c in S.CKPT if c <= STEPS} | {STEPS})
    rec = dict(run=run, src=KEEP[run], ckpt=[])

    def measure(t):
        if os.environ.get("DS_SAVE_CKPT"):  # PREREG10 R2: keep intermediate checkpoints
            torch.save(student.state_dict(), os.path.join(S.OUT, f"{run}_ck{t}.pt"))
        top1, k = evaluate(student, nrm, xh, yh, lp_t)
        rec["ckpt"].append(dict(step=t, top1=top1, kl_teacher=k))
        print(run, "step", t, "top1 %.4f kl %.4f" % (top1, k), flush=True)

    measure(0)
    for t in range(1, STEPS + 1):
        idx = torch.randint(len(xf), (S.BS,), generator=g)
        x = nrm(S.augment(xf[idx].to(S.dev, non_blocking=True), g))
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            lps = student(x[s:s + S.MB]).log_softmax(-1)
            with torch.no_grad():
                lpt = teacher(x[s:s + S.MB]).log_softmax(-1)
            ((lpt.exp() * (lpt - lps)).sum(-1).sum() / S.BS).backward()
        opt.step()
        sched.step()
        if t in ckpts:
            measure(t)
    tag = f"{run}{f'_{TRAIN}' if TRAIN != 'all' else ''}{f'_steps{STEPS}' if STEPS != S.STEPS else ''}{'_smoke' if S.SMOKE else ''}"
    json.dump(rec, open(os.path.join(S.OUT, f"{tag}.json"), "w"))
    if not S.SMOKE:
        if TRAIN.startswith("lora"):
            lora_merge(student)
        sd = student.state_dict()
        if os.environ.get("DS_SAVE_HALF"):  # PREREG24: halve disk use for the breadth models
            sd = {kk: (v.half() if v.is_floating_point() else v) for kk, v in sd.items()}
        torch.save(sd, os.path.join(S.OUT, f"{tag}.pt"))


if __name__ == "__main__":
    main(sys.argv[1])
