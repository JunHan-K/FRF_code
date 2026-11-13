"""Helpers for CaiT-style models: run the remaining blocks and the class-attention head from hidden states; KL."""

import torch




def prefix(m, x, k):
    h = m.pos_drop(m.patch_embed(x) + m.pos_embed)
    for b in m.blocks[:k + 1]:
        h = b(h)
    return h


def tail(m, h, j):
    for b in m.blocks[j + 1:]:
        h = b(h)
    c = m.cls_token.expand(h.shape[0], -1, -1)
    for b in m.blocks_token_only:
        c = b(h, c)
    return m.forward_head(m.norm(torch.cat((c, h), 1))).float().log_softmax(-1)


def kl(lp0, lp):
    return (lp0.exp() * (lp0 - lp)).sum(-1).mean()
