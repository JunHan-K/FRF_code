"""PREREG69: FRF on architectures whose blocks are not the 12 plain ViT blocks: BEiT-B (12 blocks, per-block relative
position bias), CaiT-XXS24 (24 self-attention blocks before the class-attention blocks), Swin-T stage 3 (6 blocks of width
384 on a 14x14 grid, alternating regular / shifted windows). One window of same-width residual blocks per model; references =
all native blocks of the window; E shuffles the first part, L the last part, N adds noise to the last part; cells = the
last part (L: moved slots only). Every effect is read with full forward passes of the native network in which only window
slot k is swapped (Skip, a native block, or the recovered block), i.e. native input and native suffix, as p57 'pri'.
  frfx.py train   <model> <L|E|N> <perm>         (KD recovery, recipe of deit_perm.main)
  frfx.py measure <model> <L|E|N> <perm> <slot>  (p57 primary-context Gram statistics, same keys)
  frfx.py replace <model> <L|E|N> <perm>         (p59: every native block in every cell slot, inside the recovered network)"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import spec  # noqa: E402

ARCH = {"beit_b": dict(name="beit_base_patch16_224.in22k_ft_in22k_in1k", L=12, e0=0, l0=6, n=6),
        "cait_xxs": dict(name="cait_xxs24_224.fb_dist_in1k", L=24, e0=12, l0=18, n=6),
        "swin_t3": dict(name="swin_tiny_patch4_window7_224.ms_in1k", L=6, e0=0, l0=3, n=3),
        # PREREG70 (12 blocks each; models trained above 224 are rebuilt at 224 with resampled position embeddings)
        "deit3_s22": dict(name="deit3_small_patch16_224.fb_in22k_ft_in1k", L=12, e0=0, l0=6, n=6),
        "flexi_s": dict(name="flexivit_small.1200ep_in1k", L=12, e0=0, l0=6, n=6, kw=dict(img_size=224)),
        "relpos_s": dict(name="vit_relpos_small_patch16_224.sw_in1k", L=12, e0=0, l0=6, n=6),
        "eva02_s": dict(name="eva02_small_patch14_336.mim_in22k_ft_in1k", L=12, e0=0, l0=6, n=6, kw=dict(img_size=224)),
        "vit_b32": dict(name="vit_base_patch32_224.augreg_in21k_ft_in1k", L=12, e0=0, l0=6, n=6)}
PERMS = {6: {"g": [2, 4, 1, 5, 3, 0], "h": [5, 1, 0, 3, 2, 4], "i": [1, 4, 0, 3, 2, 5], "j": [1, 0, 5, 4, 3, 2]},
         3: {"g": [1, 2, 0], "h": [2, 0, 1], "i": [2, 1, 0], "j": [1, 0, 2]}}
dev = S.dev
MB, RHO = 25, 0.3
OUTROOT = "out"


class Skip(nn.Module):
    def forward(self, x, *a, **k):
        return x


def load(model):
    import timm
    import huggingface_hub as hf
    from safetensors.torch import load_file
    n = ARCH[model]["name"]
    m = timm.create_model(n, pretrained=False, **ARCH[model].get("kw", {}))
    sd = load_file(hf.hf_hub_download("timm/" + n, "model.safetensors"))
    own = m.state_dict()
    if "pos_embed" in sd and sd["pos_embed"].shape != own["pos_embed"].shape:  # PREREG70: 240/336 -> 224 grid
        from timm.layers import resample_abs_pos_embed
        npt = 0 if getattr(m, "no_embed_class", False) else getattr(m, "num_prefix_tokens", 1)
        sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], m.patch_embed.grid_size, num_prefix_tokens=npt)
    extra = [k for k in sd if k not in own]  # derived buffers this timm version rebuilds itself
    assert all(k.endswith(("relative_position_index", "attn_mask")) for k in extra), extra[:5]
    miss = m.load_state_dict({k: v for k, v in sd.items() if k in own}, strict=False).missing_keys
    assert not [k for k in miss if not k.endswith(("relative_position_index", "attn_mask"))], miss[:5]
    return m.to(dev).eval()


def _holder(m, model):
    return (m.layers[2], "blocks") if model == "swin_t3" else (m, "blocks")


def window(m, model):
    o, a = _holder(m, model)
    return getattr(o, a)


def keep(model, kind, p):
    A = ARCH[model]
    kp = list(range(A["L"]))
    if kind in "LE":
        off = A["l0"] if kind == "L" else A["e0"]
        for i, s in enumerate(PERMS[A["n"]][p]):
            kp[off + i] = off + s
    return kp


def cell_slots(model, kind, p):
    A, kp = ARCH[model], keep(model, kind, p)
    return [k for k in range(A["l0"], A["L"]) if (kp[k] != k if kind == "L" else True)]


def build(model, kind, p):
    m = load(model)
    o, a = _holder(m, model)
    w = list(getattr(o, a))
    mods = [w[i] for i in keep(model, kind, p)]
    setattr(o, a, nn.Sequential(*mods) if isinstance(getattr(o, a), nn.Sequential) else nn.ModuleList(mods))
    return m


def nofuse(m):
    for q in m.modules():
        if hasattr(q, "fused_attn"):
            q.fused_attn = False
    return m


def run_dir(model, seed):
    return f"{OUTROOT}/s12x_{model}_seed{seed}"


def frozen(m):
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def parent(model, kind, p):
    """recovered network of `train model kind q`; p = q (seed DS_SEED, default 0) or q_seed<n> (that seed's run)"""
    q, _, sd = p.partition("_")
    seed = int(sd[4:]) if sd.startswith("seed") else S.SEED
    m = build(model, kind, q)
    m.load_state_dict(torch.load(f"{run_dir(model, seed)}/D_{kind}{q}.pt", map_location=dev))
    return frozen(m)


@torch.no_grad()
def kl_to(teacher, m, nrm, xh):
    m.eval()
    out = 0.0
    for s in range(0, len(xh), 128):
        x = nrm(xh[s:s + 128].to(dev).float().div_(255))
        lt, ls = teacher(x).float().log_softmax(-1), m(x).float().log_softmax(-1)
        out += float((lt.exp() * (lt - ls)).sum(-1).sum())
    return out / len(xh)


def train(model, kind, p):
    A = ARCH[model]
    out = run_dir(model, S.SEED)
    os.makedirs(out, exist_ok=True)
    run = f"D_{kind}{p}"
    steps = int(os.environ.get("DS_STEPS", S.STEPS))
    torch.manual_seed(S.SEED)
    teacher = frozen(load(model))
    nrm = S.Norm(teacher).to(dev)
    xh, yh = S.cache("id_hold")
    student = build(model, kind, p)
    rec = dict(run=run, model=model, src=keep(model, kind, p), ckpt=[])
    if kind == "N":  # noise on the last part, strength = median step-0 KL of the untrained L g..j networks (PREREG68 rule)
        # label-free damage matching: only KL between network outputs on id_hold images; no labels or accuracies enter
        k0 = [kl_to(teacher, build(model, "L", q), nrm, xh[:512]) for q in "ghij"]
        target = float(np.median(k0))
        w = window(student, model)
        clean = {(b, n): t.detach().clone() for b in range(A["l0"], A["L"]) for n, t in w[b].named_parameters()}
        gen = torch.Generator().manual_seed(100 + "abcdefghijklmnopqr".index(p))
        eps = {key: torch.randn(t.shape, generator=gen).to(t.device) * t.float().std() for key, t in clean.items()
               if t.is_floating_point() and t.numel() > 1}

        def set_sigma(sig):
            with torch.no_grad():
                for (b, n), t in clean.items():
                    dict(w[b].named_parameters())[n].copy_(t + sig * eps[(b, n)] if (b, n) in eps else t)

        lo, hi = math.log(1e-3), math.log(10.0)
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            set_sigma(math.exp(mid))
            if kl_to(teacher, student, nrm, xh[:512]) < target:
                lo = mid
            else:
                hi = mid
        sigma = math.exp(0.5 * (lo + hi))
        set_sigma(sigma)
        rec["noise"] = dict(target_kl0=target, sigma=sigma, late_kl0=k0)
        print(run, "target step-0 KL", target, "sigma", sigma, flush=True)
    xf, _ = S.cache("id_fit")
    xf = xf.pin_memory()
    with torch.no_grad():
        lp_t = torch.cat([teacher(nrm(xh[s:s + 128].to(dev).float().div_(255))).float() for s in range(0, len(xh), 128)]).log_softmax(-1)
    student.train()
    for q in student.parameters():
        q.requires_grad_(True)
    ps = list(student.parameters())
    print(run, model, "trainable params", sum(q.numel() for q in ps), flush=True)
    opt = torch.optim.AdamW(ps, lr=S.LR, weight_decay=S.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(S.SEED)
    ckpts = sorted({c for c in S.CKPT if c <= steps} | {steps})

    @torch.no_grad()
    def measure(t):
        student.eval()
        lg = torch.cat([student(nrm(xh[s:s + 128].to(dev).float().div_(255))).float() for s in range(0, len(xh), 128)])
        lp = lg.log_softmax(-1)
        top1, k = float((lg.argmax(1).cpu() == yh).float().mean()), float((lp_t.exp() * (lp_t - lp)).sum(-1).mean())
        student.train()
        rec["ckpt"].append(dict(step=t, top1=top1, kl_teacher=k))
        print(run, "step", t, "top1 %.4f kl %.4f" % (top1, k), flush=True)

    measure(0)
    for t in range(1, steps + 1):
        idx = torch.randint(len(xf), (S.BS,), generator=g)
        x = nrm(S.augment(xf[idx].to(dev, non_blocking=True), g))
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
    json.dump(rec, open(f"{out}/{run}.json", "w"))
    torch.save(student.state_dict(), f"{out}/{run}.pt")


def plant(E, k, p, gen):
    n, L, C = E.shape
    g = torch.randn(n, 1, C, generator=gen).to(E)
    G = spec.fip(E, E, p)
    b = spec.fip(E, g, p)[..., 0]
    w = torch.linalg.pinv(G, rtol=1e-12) @ b[..., None]
    r = g - (w.to(E.dtype).transpose(1, 2) @ E)
    rr = spec.fip(r, r, p)[:, 0, 0]
    ek = E[:, k:k + 1]
    ee = spec.fip(ek, ek, p)[:, 0, 0]
    s = (RHO / (1 - RHO) * ee / rr.clamp_min(1e-30)).sqrt()
    return ek + s.to(E.dtype)[:, None, None] * r


@torch.no_grad()
def measure(model, kind, p, slot):
    k = int(slot)
    q = p.partition("_")[0]
    nat = nofuse(frozen(load(model)))
    m = nofuse(parent(model, kind, p))
    nrm = S.Norm(nat).to(dev)
    w = window(nat, model)
    orig = list(w)
    L = len(orig)
    Bu = window(m, model)[k]
    skip = Skip()
    res = dict(model=model, kind=kind, perm=p, slot=k, src=keep(model, kind, q)[k], L=L, contexts=["pri"], rho=RHO)
    flat = {}
    lin_idx = [k, k - 1, 0]
    for sname, cname, n in (("calib", "id_fit", 1000), ("fresh", "fresh", None)):
        x, y = S.cache(cname)
        x, y = (x[:n], y[:n]) if n else (x, y)
        gen = torch.Generator().manual_seed(1000 + k)
        acc = {a: [] for a in ("G_common", "G_euclid", "pred", "kl_base_ref")}
        for s in range(0, len(x), MB):
            xb = nrm(x[s:s + MB].to(dev).float().div_(255))

            def run_with(mod):
                w[k] = mod
                return nat(xb).float()

            zs = run_with(skip)
            Z = torch.stack([run_with(orig[b]) for b in range(L)], 1)
            zb = run_with(Bu)
            w[k] = orig[k]
            zs, Z, zb = zs.double(), Z.double(), zb.double()
            E, eb = Z - zs[:, None], (zb - zs)[:, None]
            el = 0.5 * E[:, lin_idx[0]:lin_idx[0] + 1] + 0.3 * E[:, lin_idx[1]:lin_idx[1] + 1] + 0.2 * E[:, lin_idx[2]:lin_idx[2] + 1]
            pc = Z[:, k].softmax(-1)[:, None]
            pe = torch.full_like(pc, 1.0 / pc.shape[-1])
            for nm, pp in (("G_common", pc), ("G_euclid", pe)):
                Aa = torch.cat([E, eb, el, plant(E, k, pp, gen)], 1)
                acc[nm].append(spec.fip(Aa, Aa, pp).cpu())
            acc["pred"].append(torch.cat([zs.argmax(-1)[:, None], Z.argmax(-1), zb.argmax(-1)[:, None]], 1).cpu())
            lb = zb.log_softmax(-1)[:, None]
            acc["kl_base_ref"].append((lb.exp() * (lb - Z.log_softmax(-1))).sum(-1).cpu())
        d = {a: torch.cat(v).numpy() for a, v in acc.items()}
        if sname == "fresh":
            d = {a: (v.astype(np.float32) if v.dtype == np.float64 else v) for a, v in d.items() if a != "G_euclid"}
        d["label"] = y.numpy()
        flat.update({f"{sname}/pri/{a}": v for a, v in d.items()})
        print(model, kind, p, k, sname, len(x), flush=True)
    od = os.environ.get("P69_OUT", f"{OUTROOT}/p69x")
    os.makedirs(od, exist_ok=True)
    np.savez(f"{od}/p57_{model}_{kind}_{p}_{k}.npz", meta=np.array(repr(res)), **flat)


@torch.no_grad()
def replace(model, kind, p):
    q = p.partition("_")[0]
    nat = frozen(load(model))
    nrm = S.Norm(nat).to(dev)
    m = parent(model, kind, p)
    orig = list(window(nat, model))
    wm = window(m, model)
    L = len(orig)
    ks = cell_slots(model, kind, q)
    out = dict(meta=np.array(repr(dict(model=model, kind=kind, perm=p, slots=ks, src=[keep(model, kind, q)[k] for k in ks], L=L))))
    for n in ["id_hold", "fresh"]:
        x, y = S.cache(n)
        xs = [nrm(x[s:s + 128].to(dev).float().div_(255)) for s in range(0, len(x), 128)]
        base = [m(xb).float() for xb in xs]
        out[f"{n}/label"] = y.numpy()
        out[f"{n}/pred_m"] = torch.cat([z.argmax(-1) for z in base]).cpu().numpy()
        for k in ks:
            preds, kls = [], []
            own = wm[k]
            for b in range(L):
                wm[k] = orig[b]
                pr, kl = [], []
                for xb, zb in zip(xs, base):
                    lz, lb = m(xb).float().log_softmax(-1), zb.log_softmax(-1)
                    pr.append(lz.argmax(-1))
                    kl.append((lb.exp() * (lb - lz)).sum(-1))
                preds.append(torch.cat(pr).cpu())
                kls.append(torch.cat(kl).cpu())
            wm[k] = own
            out[f"{n}/{k}/pred_rep"] = torch.stack(preds, 1).numpy().astype(np.int16)
            out[f"{n}/{k}/kl_rep"] = torch.stack(kls, 1).numpy().astype(np.float32)
            print(model, kind, p, n, k, flush=True)
    od = os.environ.get("P69_REPOUT", f"{OUTROOT}/p69xrep")
    os.makedirs(od, exist_ok=True)
    np.savez(f"{od}/p59_{model}_{kind}_{p}.npz", **out)


@torch.no_grad()
def native(model):  # PREREG70 descriptive: native top-1 on id_hold at 224 (models rebuilt from 240/336)
    m = load(model)
    nrm = S.Norm(m).to(dev)
    xh, yh = S.cache("id_hold")
    pr = torch.cat([m(nrm(xh[s:s + 128].to(dev).float().div_(255))).argmax(-1).cpu() for s in range(0, len(xh), 128)])
    print(model, "native_top1", float((pr == yh).float().mean()), flush=True)


if __name__ == "__main__":
    {"train": train, "measure": measure, "replace": replace, "native": native}[sys.argv[1]](*sys.argv[2:])
