"""PREREG11.
usage: p11.py ctrl  <model> <perm> <slot> [kd_steps]   output-distillation controls on one slot: alpha + M-C
       p11.py comp  <model> <perm>[_seed1]             component-wise Block Rewind (no training)
       p11.py refit <model> <perm> <cond>              rewind slots 6..11, then KD-train only <cond> params of 6..11
                                                      cond: affine (norms+biases+layer-scale) | ln (norms) | full"""
import copy
import json
import math
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p9lib as P  # noqa: E402
import p10  # noqa: E402
from stitch import kl  # noqa: E402

OUT = "out/p11"
NA = 500
LATE = range(6, 12)


def group(n):
    """component group of a parameter name inside a block"""
    if n.startswith("norm"):
        return "norm"
    if n.startswith("ls"):
        return "ls"
    return n.split(".")[0]  # attn | mlp


def is_affine(n):
    return group(n) in ("norm", "ls", "gamma") or n.endswith(".bias")  # gamma: ConvNeXt layer scale (PREREG19)


def lora_targets(b):
    """(parent, attribute) pairs that get rank-r edits: attn.qkv/proj when the block has attention, mlp.fc1/fc2"""
    out = [(b.attn, "qkv"), (b.attn, "proj")] if hasattr(b, "attn") else []
    return out + [(b.mlp, "fc1"), (b.mlp, "fc2")]


def load_case(model, arg):
    D.M = e13.MODELS[model]
    parts = arg.split("_")
    e13.SEED_SFX = "_" + parts[1] if len(parts) > 1 else ""
    run = f"D_L{parts[0]}"
    pt, _, rj = e13.paths(model, run)
    return run, pt, json.load(open(rj))["ckpt"][0]["top1"]


def ctrl(model, p, slot, steps="1000"):
    steps = int(steps)
    os.makedirs(OUT, exist_ok=True)
    k = int(slot)
    run, pt, _ = load_case(model, p)
    i, j = D.KEEP[run][k], k
    nat = P.nofuse(S.load(D.M).eval())
    for q in nat.parameters():
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    m = P.nofuse(D.build(run).eval())
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    for q in m.parameters():
        q.requires_grad_(False)
    xf, _ = S.cache("id_fit")
    xf = torch.cat([nrm(xf[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, 2000, 100)])
    xh, _ = S.cache("id_hold")
    xh = torch.cat([nrm(xh[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, NA, 100)])
    base = m.blocks[k]
    g = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def logp(blk):
        cur = m.blocks[k]
        m.blocks[k] = blk
        lp = torch.cat([m(xh[s:s + 100].to(S.dev)).float().log_softmax(-1) for s in range(0, NA, 100)])
        m.blocks[k] = cur
        return lp

    def train_kd(target_blk):  # identical to PREREG9 A+/A- except for the step count
        blk = copy.deepcopy(base)
        for q in blk.parameters():
            q.requires_grad_(True)
        lt = logp(target_blk)
        before = float(kl(lt, logp(blk)))
        m.blocks[k] = blk
        opt = torch.optim.Adam(blk.parameters(), lr=1e-4)
        for it in range(steps):
            x = xf[torch.randint(len(xf), (32,), generator=g)].to(S.dev)
            with torch.no_grad():
                h = e13.emb(m, x)
                for b in m.blocks[:k]:
                    h = b(h)
                zt = P.run_from(m, target_blk(h), k + 1).log_softmax(-1)
            zs = P.tail_fn(m, blk(h), k + 1).log_softmax(-1)
            loss = (zt.exp() * (zt - zs)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m.blocks[k] = base
        blk.eval()
        for q in blk.parameters():
            q.requires_grad_(False)
        return blk, before, float(kl(lt, logp(blk)))

    res = dict(model=model, perm=p, slot=k, i=int(i), j=int(j), kd_steps=steps, controls={})
    lp_i, lp_j = logp(nat.blocks[i]), logp(nat.blocks[j])
    for name, tgt, truth in (("kd_neg", nat.blocks[i], 0), ("kd_pos", nat.blocks[j], 1), ("base", None, None)):
        if tgt is None:
            blk, bf, af = base, float("nan"), float("nan")
        else:
            blk, bf, af = train_kd(tgt)
        m.blocks[k] = blk
        with torch.no_grad():
            num = 0.0
            for s in range(0, NA, 50):
                h = e13.emb(m, xh[s:s + 50].to(S.dev))
                for bb in m.blocks[:k]:
                    h = bb(h)
                d0 = (nat.blocks[i](h) - h).flatten(1)
                t = (nat.blocks[j](h) - h).flatten(1) - d0
                num += float(((blk(h) - h).flatten(1) - d0).mul(t).sum(1).div(t.pow(2).sum(1).clamp_min(1e-12)).sum())
        m.blocks[k] = base
        lp = logp(blk)
        c_i, c_j = float(kl(lp, lp_i)), float(kl(lp, lp_j))
        r = dict(truth=truth, kd_kl_before=bf, kd_kl_after=af, alpha=num / NA, c_i=c_i, c_j=c_j, mc=c_i - c_j)
        res["controls"][name] = r
        print(model, p, k, steps, name, {a: round(b, 4) if isinstance(b, float) else b for a, b in r.items()}, flush=True)
    json.dump(res, open(f"{OUT}/ctrl_{model}_{p}_{k}_s{steps}.json", "w"), indent=1)


def comp(model, arg):
    os.makedirs(OUT, exist_ok=True)
    run, pt, a0 = load_case(model, arg)
    src = D.KEEP[run]
    nat = S.load(D.M).eval()
    nrm = S.Norm(nat).to(S.dev)
    xh, yh = S.cache("id_hold")
    m = D.build(run).eval()
    m.load_state_dict(torch.load(pt, map_location=S.dev))
    sdT = {kk: v.clone() for kk, v in m.state_dict().items()}
    aT = p10.top1(m, nrm, xh, yh)
    names = [n for n, _ in nat.blocks[0].named_parameters()]
    sel = {"norm": lambda n: group(n) == "norm", "attn": lambda n: group(n) == "attn",
           "mlp": lambda n: group(n) == "mlp", "ls": lambda n: group(n) == "ls",
           "bias": lambda n: n.endswith(".bias"), "affine": is_affine, "nonaffine": lambda n: not is_affine(n),
           "all": lambda n: True}
    r = dict(model=model, case=arg, top1_t0=a0, top1_T=aT, rewind={}, rel_change={})
    for gname, f in sel.items():
        chosen = [n for n in names if f(n)]
        if not chosen:
            continue
        sd = {kk: v.clone() for kk, v in sdT.items()}
        num = den = 0.0
        for k in LATE:
            nsd = nat.blocks[src[k]].state_dict()
            for n in chosen:
                num += float((sdT[f"blocks.{k}.{n}"] - nsd[n]).float().pow(2).sum())
                den += float(nsd[n].float().pow(2).sum())
                sd[f"blocks.{k}.{n}"] = nsd[n].clone()
        m.load_state_dict(sd)
        a = p10.top1(m, nrm, xh, yh)
        r["rewind"][gname] = dict(top1=a, retained=(a - a0) / (aT - a0))
        r["rel_change"][gname] = math.sqrt(num / max(den, 1e-12))
        print(model, arg, gname, r["rewind"][gname], "rel_change %.4f" % r["rel_change"][gname], flush=True)
    json.dump(r, open(f"{OUT}/comp_{model}_{arg}.json", "w"), indent=1)


class LoRA(torch.nn.Module):
    """PREREG13 H1: frozen linear + rank-r additive edit B @ A (B zero-init, so training starts at the rewound net)"""

    def __init__(self, lin, r):
        super().__init__()
        self.lin = lin
        self.A = torch.nn.Parameter(torch.randn(r, lin.in_features, device=lin.weight.device) / math.sqrt(lin.in_features))
        self.B = torch.nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device))

    def forward(self, x):
        return self.lin(x) + (x @ self.A.t()) @ self.B.t()


def refit(model, p, cond, steps=1000):
    steps = int(steps)
    os.makedirs(OUT, exist_ok=True)
    run, pt, a0 = load_case(model, p)
    src = D.KEEP[run]
    torch.manual_seed(0)
    teacher = S.load(D.M).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    nrm = S.Norm(teacher).to(S.dev)
    xh, yh = S.cache("id_hold")
    xf, _ = S.cache("id_fit")
    xf = xf.pin_memory()
    student = D.build(run)
    student.load_state_dict(torch.load(pt, map_location=S.dev))
    aT = p10.top1(student, nrm, xh, yh)
    student.load_state_dict(p10.rewound(student, teacher, src, LATE))
    a_rw = p10.top1(student, nrm, xh, yh)
    if cond.startswith("lora"):  # PREREG13 H1: rank-r edits of attn/mlp weight matrices in slots 6..11
        for k in LATE:
            for parent, n in lora_targets(student.blocks[k]):
                setattr(parent, n, LoRA(getattr(parent, n), int(cond[4:])))
    f = {"affine": is_affine, "ln": lambda n: group(n) == "norm", "full": lambda n: True,
         "affine_ctx": is_affine}.get(cond, lambda n: n.endswith(".A") or n.endswith(".B"))
    where = range(0, 6) if cond == "affine_ctx" else LATE  # PREREG12 F1: same budget, wrong place
    ps = []
    for n, q in student.named_parameters():
        on = n.startswith("blocks.") and int(n.split(".")[1]) in where and f(n.split(".", 2)[2])
        q.requires_grad_(on)
        if on:
            ps.append(q)
    lr, wd = (S.LR * 10, 0.0) if cond in ("affine", "ln", "affine_ctx") or cond.startswith("lora") else (S.LR, S.WD)  # Phase 3 convention
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(0)
    curve = {0: a_rw}
    student.train()
    for t in range(1, steps + 1):
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
        if t in (100, 300, steps):
            curve[t] = p10.top1(student, nrm, xh, yh)
            student.train()
    r = dict(model=model, perm=p, cond=cond, n_params=sum(q.numel() for q in ps), top1_t0=a0, top1_T=aT,
             top1_rewound=a_rw, curve=curve, retained={t: (a - a0) / (aT - a0) for t, a in curve.items()})
    tag = f"{cond}{f'_s{steps}' if steps != 1000 else ''}{os.environ.get('P11_TAGSFX', '')}"
    if os.environ.get("P11_SAVE"):  # PREREG15: keep the solution's slots 6..11 (LoRA folded in) for p14 measures
        sv = {}
        for k in LATE:
            b = copy.deepcopy(student.blocks[k])
            for parent, n in lora_targets(b):
                mod = getattr(parent, n)
                if isinstance(mod, LoRA):
                    with torch.no_grad():
                        mod.lin.weight += mod.B @ mod.A
                    setattr(parent, n, mod.lin)
            sv[k] = b.state_dict()
        torch.save(sv, os.environ["P11_SAVE"])
    if os.environ.get("P11_ALPHA"):  # PREREG13 H2: role measure of the refit solution
        student.eval()
        al = p10.alpha(student, teacher, nrm, xh, src)
        r.update(alpha=al, alpha_median=float(sorted(al.values())[len(al) // 2]) if al else None)
        tag += "_a"
    print(r, flush=True)
    json.dump(r, open(f"{OUT}/refit_{model}_{p}_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    {"ctrl": ctrl, "comp": comp, "refit": refit}[sys.argv[1]](*sys.argv[2:])
