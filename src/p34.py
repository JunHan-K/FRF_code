"""PREREG35: leave-one-out role spectrum of the blocks of an UNSHUFFLED pretrained ViT (no training).
For slot k (1..10), input h = native state entering block k, origin = block k skipped:
  e_b = cen(suffix_k(B_b(h))) - cen(suffix_k(h))   for every other native block b != k (references, placed at slot k)
  e_k = cen(suffix_k(B_k(h))) - cen(suffix_k(h))   (the block's own effect)
  e_k = s * sum_b w_b e_b + r (w >= 0, sum w = 1) in the Fisher metric of the full model's prediction (spec.py).
Outputs per slot: composition w, strength s, unique ratio nu (= novelty w.r.t. the other blocks), own Fisher energy,
unique energy = nu * energy, identifiability groups, and per-image values on the image and its horizontal flip.
Images: id_fit[:500] (not an evaluation set).
usage: p34.py <model>      (P34_CHECK=1: own block kept among the references, known answer w_k = 1, nu = 0)"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import e13  # noqa: E402
import s12r  # noqa: E402
import spec  # noqa: E402

OUT = "out/p34"
NA, MB = 500, int(os.environ.get("P34_MB", 25))
CHECK = os.environ.get("P34_CHECK", "0").lower() not in ("", "0", "false", "no")


def cen(z):
    return z - z.mean(-1, keepdim=True)


@torch.no_grad()
def main(model):
    os.makedirs(OUT, exist_ok=True)
    nat = S.load(s12r.MODELS[model]).eval()
    for q in nat.parameters():
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    xf, yf = S.cache("id_fit")
    xf, yf = xf[:NA], yf[:NA]
    L = len(nat.blocks)

    def suffix(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, n=NA, labels=[int(v) for v in yf], slots={})
    pred, ent = [], []
    for k in range(1, L - 1):
        refs = [b for b in range(L) if b != k or CHECK]
        pieces = {0: {"G": [], "c": [], "u": []}, 1: {"G": [], "c": [], "u": []}}
        for view in (0, 1):  # 1 = horizontal flip (reliability of the per-image values)
            for s in range(0, NA, MB):
                x = nrm(xf[s:s + MB].to(S.dev).float().div(255))
                if view:
                    x = x.flip(-1)
                h = e13.emb(nat, x)
                for b in nat.blocks[:k]:
                    h = b(h)
                z0 = cen(suffix(h, k))
                zk = suffix(nat.blocks[k](h), k)
                if k == 1 and not view:
                    pred.append(zk.argmax(-1).cpu())
                    lp = zk.log_softmax(-1)
                    ent.append(-(lp.exp() * lp).sum(-1).cpu())
                ek = (cen(zk) - z0)[:, None]
                E = torch.stack([cen(suffix(nat.blocks[b](h), k)) - z0 for b in refs], 1)
                p = zk.softmax(-1)[:, None]
                pieces[view]["G"].append(spec.fip(E, E, p).cpu())
                pieces[view]["c"].append(spec.fip(E, ek, p)[..., 0].cpu())
                pieces[view]["u"].append(spec.fip(ek, ek, p)[:, 0, 0].cpu())
        Gx, cx, ux = (torch.cat(pieces[0][t]) for t in ("G", "c", "u"))
        A = spec.analyze(Gx, cx, ux, refs)
        pi = A.pop("pi")
        F = spec.fit(*(torch.cat(pieces[1][t]) for t in ("G", "c", "u")))
        energy = float(ux.mean())
        nb = [refs.index(b) for b in (k - 1, k + 1) if b in refs]
        A.update(slot=k, refs=refs, energy=energy, unique_energy=A["nu"] * energy, unique_energy_span=A["nu_span"] * energy,
                 neighbour_share=float(sum(A["w"][t] for t in nb)),
                 pi_nu_median=float(pi["nu"].median()),
                 pi=dict(w=[[round(float(v), 3) for v in row] for row in pi["w"]], s=[round(float(v), 4) for v in pi["s"]],
                         nu=[round(float(v), 4) for v in pi["nu"]],
                         nu_span=[round(float(v), 4) for v in pi["nu_span"]], energy=[round(float(v), 5) for v in ux]),
                 flip=dict(w=[[round(float(v), 3) for v in row] for row in F["w"]], s=[round(float(v), 4) for v in F["s"]],
                           nu=[round(float(v), 4) for v in F["nu"]],
                           nu_span=[round(float(v), 4) for v in F["nu_span"]]))
        res["slots"][k] = A
        print(model, k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in A.items()
                         if kk in ("nu", "s", "energy", "cond", "neighbour_share", "groups", "group_w")}, flush=True)
    res["pred"] = [int(v) for v in torch.cat(pred)]
    res["entropy"] = [round(float(v), 4) for v in torch.cat(ent)]
    json.dump(res, open(f"{OUT}/loo_{model}{'_check' if CHECK else ''}.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
