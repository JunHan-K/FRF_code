"""PREREG47 (E1): train an early-shuffled network (D_Ea / D_Eb) with the same KD recipe as the late-shuffled ones
(src/deit_perm.py; src/s12r.py trains the late-shuffled ones). Output: out/s12r_<model>/D_E<p>.pt.
PREREG49 (E1 replication):
  seed replicate : DS_SEED=1 -> out/s12r_<model>_seed1/D_E<p>.pt (data order and augmentation only, as PREREG3)
  noise control  : run D_N<p> (p = a | b): blocks 6..11 stay in place; every parameter tensor of blocks 6..11 gets
                   Gaussian noise sigma * std(tensor) (generator seed 100 + index of p); sigma is set by bisection on
                   log sigma so that the step-0 KL to the teacher on id_hold[:512] equals the median step-0 KL of the
                   model's late-shuffled runs D_La..D_Lf (training json); then the same KD recovery.
PREREG51 (trajectories): third argument 'late' trains the late-shuffled D_L<p> (same recipe); env P47_OUTDIR redirects the
  output directory (used with DS_SAVE_CKPT=1 to keep the step 0 / 100 / 300 / 1000 checkpoints of a re-run).
PREREG58: perm c / d for E (s12r registers D_Ec / D_Ed) and N (noise generator seed 100 + index in 'abcd');
  third argument 'unperturbed' trains D_U<p>: the original network, no shuffle and no noise, same recovery recipe.
usage: p47train.py <model> <a|b|c|d> [noise|late|unperturbed]"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import s12r  # noqa: E402

model, p = sys.argv[1:3]
noise = len(sys.argv) > 3 and sys.argv[3] == "noise"
late = len(sys.argv) > 3 and sys.argv[3] == "late"
unp = len(sys.argv) > 3 and sys.argv[3] == "unperturbed"
D.M = s12r.MODELS[model]
sfx = f"_seed{S.SEED}" if S.SEED else ""
S.OUT = os.environ.get("P47_OUTDIR", f"out/s12r_{model}{sfx}")
os.makedirs(S.OUT, exist_ok=True)
run = f"D_N{p}" if noise else f"D_L{p}" if late else f"D_U{p}" if unp else f"D_E{p}"
if unp:
    D.KEEP[run] = list(range(12))

if noise:
    D.KEEP[run] = list(range(12))
    k0 = []
    for q in "abcdef":  # step-0 KL of the late-shuffled runs D_La..D_Lf (src/s12r.py), the target of the noise level
        f = f"out/s12r_{model}/D_L{q}.json"
        if os.path.exists(f):
            k0.append(json.load(open(f))["ckpt"][0]["kl_teacher"])
    if not k0:
        sys.exit(f"noise control needs the late-shuffled runs out/s12r_{model}/D_L?.json (src/s12r.py {model} D_La ...)")
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else 0.5 * sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1])  # noqa: E731
    target = med(k0) if k0 else None
    teacher = S.load(D.M).eval()
    nrm = S.Norm(teacher).to(S.dev)
    xh, _ = S.cache("id_hold")
    xh = xh[:512]

    @torch.no_grad()
    def kl_of(m):  # label-free damage matching (KL to the teacher); no id_hold labels or accuracies are used
        m.eval()
        out = 0.0
        for s in range(0, len(xh), 128):
            x = nrm(xh[s:s + 128].to(S.dev).float().div_(255))
            lt, ls = teacher(x).float().log_softmax(-1), m(x).float().log_softmax(-1)
            out += float((lt.exp() * (lt - ls)).sum(-1).sum())
        return out / len(xh)

    if target is None:  # PREREG68: models without seed-0 L a..f runs -> median step-0 KL of the untrained L<q> networks
        k0 = [kl_of(D.build(f"D_L{q}").to(S.dev)) for q in os.environ.get("P47_KL0_PERMS", "ghij")]
        target = med(k0)
    clean = {n: t.detach().clone() for n, t in teacher.state_dict().items() if any(n.startswith(f"blocks.{k}.") for k in range(6, 12))}
    gen = torch.Generator().manual_seed(100 + "abcdefghijklmnopqr".index(p))  # PREREG65-67 extend to g..r
    eps = {n: torch.randn(t.shape, generator=gen).to(t.device) * t.float().std() for n, t in clean.items()
           if t.is_floating_point() and t.numel() > 1}

    _build = D.build

    def noised(sigma):
        m = _build(run)
        sd = m.state_dict()
        for n, e in eps.items():
            sd[n] = clean[n] + sigma * e.to(sd[n].dtype)
        m.load_state_dict(sd)
        return m

    lo, hi = math.log(1e-3), math.log(10.0)
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if kl_of(noised(math.exp(mid)).to(S.dev)) < target:
            lo = mid
        else:
            hi = mid
    sigma = math.exp(0.5 * (lo + hi))
    print(run, "target step-0 KL", target, "sigma", sigma, flush=True)
    json.dump(dict(target_kl0=target, sigma=sigma, late_kl0=k0), open(f"{S.OUT}/{run}_noise.json", "w"))
    D.build = lambda r, _f=noised, _s=sigma: _f(_s)
    del teacher

if not os.path.exists(f"{S.OUT}/{run}.pt"):
    D.main(run)
