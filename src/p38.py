"""PREREG38: block substitution in an UNSHUFFLED pretrained ViT (no training). Does the FRF direction predict what the
magnitude cannot?
Slot k (1..10), input h = native state entering block k, origin = block k skipped, reference axis = native block B_k:
  e_b = z(B_b at slot k) - z(skip),  e_k = z(native) - z(skip),  Fisher metric of the native prediction p (per image).
Predictors (id_fit[:500], pooled over images):
  frf_cos = <e_b, e_k>_F / (|e_b|_F |e_k|_F),  frf_R = <e_b, e_k>_F / |e_k|^2_F,  gamma_star = <e_b, e_k>_F / |e_b|^2_F,
  frf_dist = |e_b - e_k|^2_F / |e_k|^2_F (magnitude),  lens_cos (same, slot output read by final norm + head),
  local_cos (hidden deltas B_b(h) - h vs B_k(h) - h, all tokens),  cka (linear CKA of the CLS outputs).
Truth (id_hold, 1000): top-1 / KL-to-native of the net with h + gamma (B_b(h) - h) at slot k, gamma in GAMMAS.
Known-answer candidates: "self" (B_k) and "half" (h + 0.5 (B_k(h) - h)).
usage: p38.py <model> [comma-separated slots]"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
import surgery as S  # noqa: E402
import e13  # noqa: E402
import s12r  # noqa: E402

OUT = "out/p38"
NA, NH, MB = 500, 1000, int(os.environ.get("P38_MB", 50))
GAMMAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def fcov(a, b, p):
    """Fisher inner product Cov_p(a, b) per image, a/b/p (N, C) -> (N,) float64"""
    return ((p * a * b).sum(-1) - (p * a).sum(-1) * (p * b).sum(-1)).double()


def lin_cka(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    return float((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm()))


def norm_imgs(nrm, x):
    return torch.cat([nrm(x[s:s + 100].to(S.dev).float().div_(255)).cpu() for s in range(0, len(x), 100)])


@torch.no_grad()
def main(model, slots=None):
    os.makedirs(OUT, exist_ok=True)
    nat = S.load(s12r.MODELS[model]).eval()
    for q in nat.parameters():
        q.requires_grad_(False)
    nrm = S.Norm(nat).to(S.dev)
    L = len(nat.blocks)
    ks = [int(t) for t in slots.split(",")] if slots else list(range(1, L - 1))
    xf, _ = S.cache("id_fit")
    xf = norm_imgs(nrm, xf[:NA])
    xh, yh = S.cache("id_hold")
    xh, yh = norm_imgs(nrm, xh[:NH]), yh[:NH].to(S.dev)

    def prefix(x, k):
        h = e13.emb(nat, x)
        for b in nat.blocks[:k]:
            h = b(h)
        return h

    def suffix(o, k):
        for b in nat.blocks[k + 1:]:
            o = b(o)
        return nat.forward_head(nat.norm(o)).float()

    def lens(o):
        return nat.forward_head(nat.norm(o)).float()

    res = dict(model=model, gammas=GAMMAS, n_fit=NA, n_hold=NH, slots={})
    for k in ks:
        Bk = nat.blocks[k]
        cands = [b for b in range(L) if b != k] + ["self", "half"]

        def f(c, h):
            if c == "self":
                return Bk(h)
            if c == "half":
                return h + 0.5 * (Bk(h) - h)
            return nat.blocks[c](h)

        # ---------- predictors on id_fit ----------
        acc = {c: dict(fbk=0.0, fbb=0.0, lbk=0.0, lbb=0.0, dbk=0.0, dbb=0.0, cls=[]) for c in cands}
        fkk = lkk = dkk = 0.0
        clsk = []
        for s in range(0, NA, MB):
            h = prefix(xf[s:s + MB].to(S.dev), k)
            ok_ = Bk(h)
            zs, zk = suffix(h, k), suffix(ok_, k)
            p = zk.softmax(-1)
            ek = zk - zs
            ls, lk = lens(h), lens(ok_)
            pl = lk.softmax(-1)
            elk = lk - ls
            dk = (ok_ - h).flatten(1).double()
            fkk += float(fcov(ek, ek, p).sum())
            lkk += float(fcov(elk, elk, pl).sum())
            dkk += float((dk * dk).sum())
            clsk.append(ok_[:, 0].double().cpu())
            for c in cands:
                ob = f(c, h)
                eb, elb, db = suffix(ob, k) - zs, lens(ob) - ls, (ob - h).flatten(1).double()
                a = acc[c]
                a["fbk"] += float(fcov(eb, ek, p).sum())
                a["fbb"] += float(fcov(eb, eb, p).sum())
                a["lbk"] += float(fcov(elb, elk, pl).sum())
                a["lbb"] += float(fcov(elb, elb, pl).sum())
                a["dbk"] += float((db * dk).sum())
                a["dbb"] += float((db * db).sum())
                a["cls"].append(ob[:, 0].double().cpu())
        Ck = torch.cat(clsk)
        pred = {}
        for c in cands:
            a = acc[c]
            pred[str(c)] = dict(
                frf_cos=a["fbk"] / max((a["fbb"] * fkk) ** 0.5, 1e-30), frf_R=a["fbk"] / max(fkk, 1e-30),
                gamma_star=a["fbk"] / max(a["fbb"], 1e-30), frf_dist=(a["fbb"] - 2 * a["fbk"] + fkk) / max(fkk, 1e-30),
                frf_energy_ratio=a["fbb"] / max(fkk, 1e-30),
                lens_cos=a["lbk"] / max((a["lbb"] * lkk) ** 0.5, 1e-30),
                local_cos=a["dbk"] / max((a["dbb"] * dkk) ** 0.5, 1e-30), cka=lin_cka(torch.cat(a["cls"]), Ck))
        # ---------- truth on id_hold ----------
        t_nat = t_skip = 0
        cor = {str(c): {g: 0 for g in GAMMAS} for c in cands}
        klv = {str(c): {g: 0.0 for g in GAMMAS} for c in cands}
        for s in range(0, NH, MB):
            h = prefix(xh[s:s + MB].to(S.dev), k)
            y = yh[s:s + MB]
            zn = suffix(Bk(h), k)
            lpn = zn.log_softmax(-1)
            t_nat += int((zn.argmax(-1) == y).sum())
            t_skip += int((suffix(h, k).argmax(-1) == y).sum())
            for c in cands:
                d = f(c, h) - h
                for g in GAMMAS:
                    z = suffix(h + g * d, k)
                    cor[str(c)][g] += int((z.argmax(-1) == y).sum())
                    klv[str(c)][g] += float((lpn.exp() * (lpn - z.log_softmax(-1))).sum(-1).sum())
        top1 = {c: {str(g): v / NH for g, v in d.items()} for c, d in cor.items()}
        kls = {c: {str(g): v / NH for g, v in d.items()} for c, d in klv.items()}
        res["slots"][k] = dict(top1_nat=t_nat / NH, top1_skip=t_skip / NH, pred=pred, top1=top1, kl=kls)
        print(model, k, "nat", t_nat / NH, "skip", t_skip / NH,
              {c: (round(pred[c]["frf_cos"], 3), round(max(top1[c].values()), 3), round(top1[c]["1.0"], 3)) for c in pred}, flush=True)
    tag = f"_{slots.replace(',', '-')}" if slots else ""
    json.dump(res, open(f"{OUT}/p38_{model}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
