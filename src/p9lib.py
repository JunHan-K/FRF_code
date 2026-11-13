"""PREREG9 helpers: output-effect signatures (exact skip effect and first-order jvp) and task-loss stitching."""
import torch

import e13

dev = "cuda"


def nofuse(m):
    for b in m.blocks:
        if hasattr(b, "attn"):  # PREREG19: ConvNeXt blocks have no attention
            b.attn.fused_attn = False
    return m


def tail_fn(m, h, start):
    for b in m.blocks[start:]:
        h = b(h)
    return m.forward_head(m.norm(h)).float()


@torch.no_grad()
def run_from(m, h, start):
    return tail_fn(m, h, start)


def effects(m, xs, k, n=500, fo=True):
    """e_k = z_full - z_skip_k  and  e_hat_k = J_{h_{k+1} -> z} Delta_k  (logits, [n, C])"""
    E, F = [], []
    for s in range(0, n, 25):
        x = xs[s:s + 25].to(dev)
        with torch.no_grad():
            h = e13.emb(m, x)
            for b in m.blocks[:k]:
                h = b(h)
            h1 = m.blocks[k](h)
            zf, zs = run_from(m, h1, k + 1), run_from(m, h, k + 1)
        E.append((zf - zs).cpu())
        if fo:
            f = lambda u: tail_fn(m, u, k + 1)
            _, j = torch.func.jvp(f, (h1,), (h1 - h,))
            F.append(j.float().detach().cpu())
    return torch.cat(E), (torch.cat(F) if fo else None)


def rowcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def eff_score(e_meas, e_j, e_i):
    return float((rowcorr(e_meas, e_j) - rowcorr(e_meas, e_i)).mean())


def tl_stitch(nat, blk, pos, xs_fit, xs_ev, lp_ev, steps=300, identity=False):
    """task-loss stitching: affine in/out adapters around blk (or identity) at native position pos, trained with the
    native model's own output distribution as target (KL). Returns eval KL."""
    d = nat.embed_dim
    A_in, A_out = torch.nn.Linear(d, d).to(dev), torch.nn.Linear(d, d).to(dev)
    for A in (A_in, A_out):
        torch.nn.init.eye_(A.weight)
        torch.nn.init.zeros_(A.bias)
    opt = torch.optim.Adam(list(A_in.parameters()) + list(A_out.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(0)

    def fwd(x):
        with torch.no_grad():
            h = e13.emb(nat, x)
            for b in nat.blocks[:pos]:
                h = b(h)
        u = A_in(h)
        v = u if identity else blk(u)
        h = A_out(v)
        for b in nat.blocks[pos + 1:]:
            h = b(h)
        return nat.forward_head(nat.norm(h)).float().log_softmax(-1)

    for it in range(steps):
        idx = torch.randint(len(xs_fit), (32,), generator=g)
        x = xs_fit[idx].to(dev)
        with torch.no_grad():
            lt = nat(x).float().log_softmax(-1)
        loss = (lt.exp() * (lt - fwd(x))).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tot = 0.0
        for s in range(0, len(xs_ev), 100):
            lp = fwd(xs_ev[s:s + 100].to(dev))
            l0 = lp_ev[s:s + 100]
            tot += float((l0.exp() * (l0 - lp)).sum(-1).sum())
    return tot / len(xs_ev)
