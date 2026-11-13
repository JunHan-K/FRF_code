"""Block removal (PREREG23 / PREREG25 C): choose k = 2, 3, 4 blocks to remove with a criterion, recover the shortened
network by distillation, and measure accuracy (ImageNet-V2 id_hold, ImageNet-R / -A) and throughput. Blocks are chosen
on id_fit images; all accuracies are measured on held-out sets. Also the shared helpers (frozen copies, top-1, distillation,
LoRA wrapping, evaluation sets) used by the other experiments.
usage: p18.py prune23 <model> <skipG|biG|BI|CKA|random|taylor> <lora1|full>
       P18_RECIPE=main (default; main-text removal) | cheap (PREREG23 / PREREG25 C); P18_SEED=1, 2 for seed replicates"""
import copy
import json
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p11  # noqa: E402
import s12r  # noqa: E402
from imagenet_subsets import IMAGENET_A, IMAGENET_R  # noqa: E402

OUT = "out/p18"
dev = S.dev


def frozen(m):
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


@torch.no_grad()
def top1(m, nrm, x, y, mask=None, bs=128):
    m.eval()
    ok = 0
    for s in range(0, len(x), bs):
        z = m(nrm(x[s:s + bs].to(dev).float().div_(255))).float()
        if mask is not None:
            z = z + mask
        ok += int((z.argmax(1).cpu() == y[s:s + bs]).sum())
    return ok / len(x)


def kd_train(student, teacher, nrm, xf, params, steps, lr, wd=0.0, seed=0, featkd=False):
    """Phase 2 KD recipe (batch 64, micro 16, same augmentation, warmup + cosine) on the given parameters only."""
    for q in student.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S.WARM) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(seed)
    student.train()
    for t in range(steps):
        idx = torch.randint(len(xf), (S.BS,), generator=g)
        x = nrm(S.augment(xf[idx].to(dev, non_blocking=True), g))
        opt.zero_grad(set_to_none=True)
        for s in range(0, S.BS, S.MB):
            if featkd:  # PREREG30: also match the final token features (after the last block's norm), scale-normalised
                fs = student.forward_features(x[s:s + S.MB])
                lps = student.forward_head(fs).log_softmax(-1)
                with torch.no_grad():
                    ft = teacher.forward_features(x[s:s + S.MB])
                    lpt = teacher.forward_head(ft).log_softmax(-1)
                floss = (fs.float() - ft.float()).pow(2).mean() / ft.float().pow(2).mean().clamp_min(1e-8)
                ((lpt.exp() * (lpt - lps)).sum(-1).sum() / S.BS + floss * len(fs) / S.BS).backward()
            else:
                lps = student(x[s:s + S.MB]).log_softmax(-1)
                with torch.no_grad():
                    lpt = teacher(x[s:s + S.MB]).log_softmax(-1)
                ((lpt.exp() * (lpt - lps)).sum(-1).sum() / S.BS).backward()
        opt.step()
        sched.step()
    student.eval()
    return student


def lora_wrap(blk, r):
    blk.attn.qkv, blk.attn.proj = p11.LoRA(blk.attn.qkv, r), p11.LoRA(blk.attn.proj, r)
    blk.mlp.fc1, blk.mlp.fc2 = p11.LoRA(blk.mlp.fc1, r), p11.LoRA(blk.mlp.fc2, r)
    return [q for n, q in blk.named_parameters() if n.endswith(".A") or n.endswith(".B")]


def affine_params(blk):
    return [q for n, q in blk.named_parameters() if p11.is_affine(n)]


def eval_sets():
    """(x, y, mask) per evaluation set; ImageNet-R / ImageNet-A are evaluated inside their official 200 classes"""
    out = {}
    for n, classes in (("id_hold", None), ("shift_R", IMAGENET_R), ("shift_A", IMAGENET_A)):
        x, y = S.cache(n)
        mask = None
        if classes is not None:
            mask = torch.full((1000,), -1e4, device=dev)
            mask[torch.tensor(classes, device=dev)] = 0.0
        out[n] = (x, y, mask)
    return out


def save(name, r):
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(f"{OUT}/{name}.json", "w"), indent=1)
    print(name, json.dumps(r)[:1500], flush=True)


def lin_cka(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    return float((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm()))


def prune23(model, crit, rec):
    """PREREG23: role-aware block-removal recipe. crit: skipG (ours, greedy output KL) | biG (greedy Block Influence)
    | BI | CKA (static single-drop ranking on the native net) | random. rec: lora1 | full.
    For k = 2, 3, 4: accuracy (V2 / R / A), trainable params, selection wall-clock, measured throughput."""
    import time
    SEED = int(os.environ.get("P18_SEED", 0))  # PREREG25 C: recovery-seed replicates (LoRA init + data order)
    # PREREG30 recovery recipe (defaults = the original PREREG23 recipe)
    # recovery recipe: "main" = the removal results of the main text (rank-4 LoRA, 3,000 steps, final-token feature KD,
    # k = 1..4); "cheap" = the rank-1 / 1,000-step recipe of PREREG23 / PREREG25 C (k = 2..4). Single values can be overridden.
    rc = os.environ.get("P18_RECIPE", "main")
    base = dict(main=("4", "3000", "1", "1,2,3,4", "_main"), cheap=("1", "1000", "0", "2,3,4", "_cheap"))[rc]
    RANK, STEPS = int(os.environ.get("P18_RANK", base[0])), int(os.environ.get("P18_STEPS", base[1]))
    FEAT = bool(int(os.environ.get("P18_FEATKD", base[2])))
    KS = tuple(int(k) for k in os.environ.get("P18_KS", base[3]).split(","))
    RTAG = os.environ.get("P18_RTAG", base[4])
    torch.manual_seed(SEED)
    D.M = s12r.MODELS[model]
    nat = frozen(S.load(D.M))
    nrm = S.Norm(nat).to(dev)
    xf, _ = S.cache("id_fit")
    xh, _ = S.cache("id_fit")  # blocks are chosen on training images; accuracy is measured on id_hold
    xn = torch.cat([nrm(xh[s:s + 100].to(dev).float().div_(255)).cpu() for s in range(0, 500, 100)])
    sets = eval_sets()
    L = len(nat.blocks)

    @torch.no_grad()
    def run_blocks(blocks, x):
        h = e13.emb(nat, x)
        for t in blocks:
            h = nat.blocks[t](h)
        return nat.forward_head(nat.norm(h)).float()

    @torch.no_grad()
    def kl_blocks(blocks):
        tot = 0.0
        for s in range(0, 500, 100):
            x = xn[s:s + 100].to(dev)
            lt, ls = nat(x).float().log_softmax(-1), run_blocks(blocks, x).log_softmax(-1)
            tot += float((lt.exp() * (lt - ls)).sum(-1).sum())
        return tot / 500

    @torch.no_grad()
    def fisher_blocks(blocks):
        """PREREG28 CFE importance: E[d^T H d], d = logits(pruned) - logits(native), H = Fisher of the native output"""
        tot = 0.0
        for s in range(0, 500, 100):
            x = xn[s:s + 100].to(dev)
            zn = nat(x).float()
            d = run_blocks(blocks, x) - zn
            p = zn.softmax(-1)
            tot += float(((p * d * d).sum(-1) - (p * d).sum(-1).pow(2)).sum())
        return tot / 500

    @torch.no_grad()
    def cls_io(blocks, pos):
        I, O = [], []
        for s in range(0, 500, 100):
            h = e13.emb(nat, xn[s:s + 100].to(dev))
            for t in blocks[:pos]:
                h = nat.blocks[t](h)
            o = nat.blocks[blocks[pos]](h)
            I.append(h[:, 0].double())
            O.append(o[:, 0].double())
        return torch.cat(I), torch.cat(O)

    cand = list(range(1, L - 1))
    t0 = time.time()
    order = []
    if crit in ("skipG", "biG", "cfeG"):
        keep = list(range(L))
        for _ in range(4):
            sc = {}
            for b in [t for t in keep if t in cand]:
                if crit == "skipG":
                    sc[b] = -kl_blocks([t for t in keep if t != b])
                elif crit == "cfeG":
                    sc[b] = -fisher_blocks([t for t in keep if t != b])
                else:
                    I, O = cls_io(keep, keep.index(b))
                    sc[b] = float(torch.nn.functional.cosine_similarity(I, O, dim=1).mean())
            b = max(sc, key=sc.get)
            keep.remove(b)
            order.append(b)
    elif crit == "taylor":  # PREREG30: first-order (Taylor) estimate of the change in the top-1 log-prob when a block is removed
        sc = {}
        for b in cand:
            tot = 0.0
            for s in range(0, 500, 100):
                x = xn[s:s + 100].to(dev)
                with torch.no_grad():
                    h = e13.emb(nat, x)
                    for t in range(b):
                        h = nat.blocks[t](h)
                    o = nat.blocks[b](h)
                    top = nat(x).argmax(-1)
                o = o.detach().requires_grad_(True)
                hh = o
                for t in range(b + 1, L):
                    hh = nat.blocks[t](hh)
                lp = nat.forward_head(nat.norm(hh)).float().log_softmax(-1).gather(1, top[:, None]).sum()
                gr = torch.autograd.grad(lp, o)[0]
                tot += float((gr * (h - o)).flatten(1).sum(1).abs().sum())
            sc[b] = -tot / 500  # small estimated change = safe to drop
        order = sorted(cand, key=lambda b: -sc[b])[:4]
    elif crit in ("BI", "CKA"):
        full = list(range(L))
        sc = {}
        for b in cand:
            I, O = cls_io(full, b)
            sc[b] = float(torch.nn.functional.cosine_similarity(I, O, dim=1).mean()) if crit == "BI" else lin_cka(I, O)
        order = sorted(cand, key=lambda b: -sc[b])[:4]  # most similar in/out (least influence / most redundant) first
    else:
        order = [cand[t] for t in torch.randperm(len(cand), generator=torch.Generator().manual_seed(0))[:4].tolist()]
    sel_time = time.time() - t0

    @torch.no_grad()
    def throughput(m):
        m.eval()
        x = torch.randn(64, 3, 224, 224, device=dev)
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            m(x)
        torch.cuda.synchronize()
        return 640 / (time.time() - t)

    res = dict(model=model, crit=crit, rec=rec, order=order, select_seconds=sel_time,
               recipe=dict(rank=RANK, steps=STEPS, featkd=FEAT, ks=KS),
               native={n: top1(nat, nrm, *v) for n, v in sets.items()}, native_ips=throughput(nat), fix=True,
               native_params=sum(q.numel() for q in nat.parameters()), steps=[])
    for k in KS:
        drop = sorted(order[:k])
        m = copy.deepcopy(nat)
        m.blocks = nn.Sequential(*[blk for t, blk in enumerate(m.blocks) if t not in drop])
        if rec == "lora1":
            ps = []
            for blk in m.blocks:
                ps += lora_wrap(blk, RANK)
            kd_train(m, nat, nrm, xf, ps, STEPS, 1e-3, seed=SEED, featkd=FEAT)
            for blk in m.blocks:  # fold the rank-1 edits before timing
                for parent, nme in p11.lora_targets(blk):
                    mod = getattr(parent, nme)
                    with torch.no_grad():
                        mod.lin.weight += mod.B @ mod.A
                    setattr(parent, nme, mod.lin)
        else:
            ps = list(m.parameters())
            kd_train(m, nat, nrm, xf, ps, STEPS, S.LR, S.WD, seed=SEED, featkd=FEAT)
        m = frozen(m)
        r = dict(k=k, drop=drop, n_trainable=sum(q.numel() for q in ps), params=sum(q.numel() for q in m.parameters()),
                 acc={n: top1(m, nrm, *v) for n, v in sets.items()}, ips=throughput(m))
        res["steps"].append(r)
        print(model, crit, rec, r, flush=True)
        del m
    save(f"p23_{model}_{crit}_{rec}{f'_seed{SEED}' if SEED else ''}_fix{RTAG}", res)


if __name__ == "__main__":
    {"prune23": prune23}[sys.argv[1]](*sys.argv[2:])
