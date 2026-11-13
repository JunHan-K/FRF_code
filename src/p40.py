"""PREREG40: how special is the functional overlap between native ViT blocks? Null references for the leave-one-out map.
For slot k (1..10), input h = native state entering block k, origin = block skipped, target e_k, Fisher metric of p_k:
  own     : references = the other native blocks of the same model (as PREREG35)
  perm    : the same references taken from a different image (image index permuted, 3 permutations)
  foreign : the blocks of an independently trained model of the same width applied to the same h (11 references)
  gauss   : random logit directions with each own reference's per-image Fisher norm (3 draws)
Per frame: shared_span = 1 - nu_span (unconstrained least squares, the reviewer's span criterion), shared_cone = 1 - nu (NNLS),
and for own: neighbour share of the NNLS composition and of the unconstrained explained energy.
Images: id_fit[:500].   usage: p40.py <model>"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import e13  # noqa: E402
import s12r  # noqa: E402
import spec  # noqa: E402

OUT = "out/p40"
NA, MB = 500, int(os.environ.get("P40_MB", 25))
FOREIGN = {"deit_s": "deit3_s", "deit3_s": "augreg_s", "augreg_s": "deit_s",
           "deit_b": "deit3_b", "deit3_b": "augreg_b", "augreg_b": "deit_b", "clip_b": "deit_b"}


def cen(z):
    return z - z.mean(-1, keepdim=True)


def frame(E, ek, p):
    """E (N,R,C) references, ek (N,C) target, p (N,C) metric -> pooled fit summary"""
    G = torch.cat([spec.fip(E[s:s + 50].to(S.dev), E[s:s + 50].to(S.dev), p[s:s + 50, None].to(S.dev)).cpu() for s in range(0, len(E), 50)])
    c = torch.cat([spec.fip(E[s:s + 50].to(S.dev), ek[s:s + 50, None].to(S.dev), p[s:s + 50, None].to(S.dev))[..., 0].cpu()
                   for s in range(0, len(E), 50)])
    u = torch.cat([spec.fip(ek[s:s + 50, None].to(S.dev), ek[s:s + 50, None].to(S.dev), p[s:s + 50, None].to(S.dev))[:, 0, 0].cpu()
                   for s in range(0, len(E), 50)])
    F = spec.fit(G.sum(0), c.sum(0), u.sum())
    return dict(shared_span=float(1 - F["nu_span"]), shared_cone=float(1 - F["nu"]), w=[round(float(x), 4) for x in F["w"]],
                wu=[round(float(x), 4) for x in F["wu"]], G=G.sum(0), c=c.sum(0), w_raw=F["w"], wu_raw=F["wu"])


@torch.no_grad()
def main(model):
    os.makedirs(OUT, exist_ok=True)
    nat = S.load(s12r.MODELS[model]).eval()
    fm = S.load(s12r.MODELS[FOREIGN[model]]).eval()
    for q in list(nat.parameters()) + list(fm.parameters()):
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    xf, _ = S.cache("id_fit")
    xf = xf[:NA]
    L = len(nat.blocks)
    g = torch.Generator().manual_seed(0)
    perms = [torch.randperm(NA, generator=g) for _ in range(3)]

    def suffix(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, foreign=FOREIGN[model], n=NA, slots={})
    for k in range(1, L - 1):
        refs = [b for b in range(L) if b != k]
        Eo, Ef, EK, PK = [], [], [], []
        for s in range(0, NA, MB):
            x = nrm(xf[s:s + MB].to(S.dev).float().div(255))
            h = e13.emb(nat, x)
            for b in nat.blocks[:k]:
                h = b(h)
            z0 = cen(suffix(h, k))
            zk = suffix(nat.blocks[k](h), k)
            EK.append((cen(zk) - z0).cpu())
            PK.append(zk.softmax(-1).cpu())
            Eo.append(torch.stack([cen(suffix(nat.blocks[b](h), k)) - z0 for b in refs], 1).cpu())
            Ef.append(torch.stack([cen(suffix(fm.blocks[b](h), k)) - z0 for b in refs], 1).cpu())
        Eo, Ef, ek, p = torch.cat(Eo), torch.cat(Ef), torch.cat(EK), torch.cat(PK)
        own = frame(Eo, ek, p)
        nb = [refs.index(b) for b in (k - 1, k + 1) if b in refs]
        own.pop("G")
        cu, wu, wr = own.pop("c"), own.pop("wu_raw").double(), own.pop("w_raw").double()  # unrounded weights
        expl = wu * cu  # per-reference share of the explained energy (unconstrained fit; sums to explained energy)
        own.update(nbr_share_cone=float(sum(wr[t] for t in nb)),
                   nbr_share_span=float(expl[nb].sum() / expl.sum().clamp_min(1e-30)) if float(expl.sum()) > 0 else None)
        nulls = {"perm": [], "foreign": [], "gauss": []}
        for pm in perms:
            f = frame(Eo[pm], ek, p)
            nulls["perm"].append(dict(shared_span=f["shared_span"], shared_cone=f["shared_cone"]))
        f = frame(Ef, ek, p)
        nulls["foreign"].append(dict(shared_span=f["shared_span"], shared_cone=f["shared_cone"]))
        for d in range(3):
            gg = torch.Generator().manual_seed(100 + d)
            Rn = torch.randn(Eo.shape, generator=gg)
            no = torch.cat([spec.fip(Eo[s:s + 50].to(S.dev), Eo[s:s + 50].to(S.dev), p[s:s + 50, None].to(S.dev)).diagonal(dim1=1, dim2=2).cpu()
                            for s in range(0, NA, 50)])
            nr = torch.cat([spec.fip(Rn[s:s + 50].to(S.dev), Rn[s:s + 50].to(S.dev), p[s:s + 50, None].to(S.dev)).diagonal(dim1=1, dim2=2).cpu()
                            for s in range(0, NA, 50)])
            Rn = Rn * (no / nr.clamp_min(1e-30)).sqrt().float()[..., None]
            f = frame(Rn, ek, p)
            nulls["gauss"].append(dict(shared_span=f["shared_span"], shared_cone=f["shared_cone"]))
        # known-answer: own block among the references -> shared_span = 1
        ka = frame(torch.cat([Eo, ek[:, None]], 1), ek, p)
        res["slots"][k] = dict(own=own, nulls=nulls, ka_self_shared_span=ka["shared_span"])
        print(model, k, "own", round(own["shared_span"], 3), round(own["shared_cone"], 3),
              {n: [round(v["shared_span"], 3) for v in vs] for n, vs in nulls.items()}, "ka", round(ka["shared_span"], 4), flush=True)
    json.dump(res, open(f"{OUT}/p40_{model}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
