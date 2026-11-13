"""PREREG85 Part 2 (GPU): cross-run Fisher Gram of recovered blocks, so that full effects and out-of-frame residuals of
different recovery runs can be compared. Written 2026-11-05 15:40 KST, before any Part 2 value.
Primary context: native input entering slot k, native suffix after k; common Fisher (p = the native network's own
prediction, native block k in slot k) -- identical to PREREG57's fresh/pri G_common, so the (references, own block)
sub-Gram of every network must reproduce the stored file (known answer).
One job = one ViT-S model. For each slot k = 6..11, candidates = block k of every seed-0 and seed-1 L network (perms
a-f) whose block k was relocated (KEEP[k] != k). Stored: G (12 + n_cand)^2 summed over the fresh set (float64), the
per-image G for the first 200 images, candidate ids (perm, seed, src).
usage: p85.py <model>"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import deit_perm as D  # noqa: E402
import e13  # noqa: E402
import p9lib as P  # noqa: E402
import p11  # noqa: E402
import spec  # noqa: E402

OUT = os.environ.get("P85_OUT", "out/p85")
MB, NKEEP = 25, 200
PERMS = [p + s for s in ("", "_seed1") for p in "abcdef"]


@torch.no_grad()
def main(model):
    os.makedirs(OUT, exist_ok=True)
    blocks, ids = {}, []
    for p in PERMS:  # keep blocks 6-11 of every network, drop the rest
        run, pt, _ = p11.load_case(model, p)
        m = P.nofuse(D.build(run).eval())
        m.load_state_dict(torch.load(pt, map_location="cpu"))
        for k in range(6, 12):
            if D.KEEP[run][k] != k:
                blocks[(p, k)] = m.blocks[k].to(S.dev).eval()
                for q in blocks[(p, k)].parameters():
                    q.requires_grad_(False)
        ids.append((p, run))
        del m
        print("loaded", model, p, flush=True)
    nat = P.nofuse(S.load(D.M).eval())
    for q in nat.parameters():
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    x, _ = S.cache("fresh")
    for k in range(6, 12):
        f = f"{OUT}/p85_{model}_{k}.npz"
        if os.path.exists(f):
            continue
        cand = [(p, run) for p, run in ids if (p, k) in blocks]
        G, Gi = 0.0, []
        for s in range(0, len(x), MB):
            xb = nrm(x[s:s + MB].to(S.dev).float().div_(255))
            h = e13.emb(nat, xb)
            for b in nat.blocks[:k]:
                h = b(h)

            def suffix(o):
                for b in nat.blocks[k + 1:]:
                    o = b(o)
                return nat.forward_head(nat.norm(o)).float()

            zs = suffix(h).double()
            Z = torch.stack([suffix(nat.blocks[r](h)) for r in range(12)], 1).double()
            C = torch.stack([suffix(blocks[(p, k)](h)) for p, _ in cand], 1).double()
            E = torch.cat([Z, C], 1) - zs[:, None]
            g = spec.fip(E, E, Z[:, k].softmax(-1)[:, None]).cpu()
            G = G + g.sum(0)
            if s < NKEEP:
                Gi.append(g)
        meta = dict(model=model, slot=k, n=len(x), cand=[dict(perm=p, seed=int("seed1" in p), src=int(D.KEEP[run][k]))
                                                         for p, run in cand])
        np.savez(f, G=G.numpy(), G_img=torch.cat(Gi)[:NKEEP].numpy(), meta=np.array(repr(meta)))
        print("saved", f, G.shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
