import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

import paths

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, CACHE = paths.DATA, paths.CACHE
OUT = os.environ.get("KT_OUT", os.path.join(ROOT, "out"))
LAYERS = ["layer2", "layer3", "layer4"]
# KT_POS=dissect (ViT only): positions inside the last block, each read out as patch mean (pm) and CLS
DISSECT = ["b11_pm", "b11_cls", "attn_pm", "attn_cls", "mlp_pm", "mlp_cls", "ln_pm", "ln_cls"]
# ResNet counterpart: layer3 output and the outputs of the three layer4 bottlenecks
RDISSECT = ["l3", "l4b0", "l4b1", "l4b2"]
if os.environ.get("KT_POS") == "dissect":
    LAYERS = DISSECT if os.environ.get("KT_ARCH", "resnet") == "vit" else RDISSECT
# KT_POS=interv (ViT only): A2 causal test. Block-12 attention is replaced per KT_INTERV:
# none (manual reimplementation), uniform (all heads attend uniformly), noattn (attention branch removed), h<k> (head k zeroed)
INTERV = ["mlp_pm", "ln_cls"]
INTERV_MODE = os.environ.get("KT_INTERV", "none")
INTERV_BLK = int(os.environ.get("KT_IBLK", 11))  # block (0-based) whose attention is edited; 11 = block 12
# KT_POS=blocks (ViT only): output of every block, read as patch mean and CLS
BLOCKS = [f"k{i}_{r}" for i in range(12) for r in ("pm", "cls")]
if os.environ.get("KT_POS") == "blocks":
    LAYERS = BLOCKS
# KT_POS=gblocks (KT_ARCH=gen, any timm model): output of each of KT_NBLK blocks, pooled over tokens/space (pm)
# and, when KT_CLS=1, the first (CLS) token
GBLOCKS = [f"k{i}_{r}" for i in range(int(os.environ.get("KT_NBLK", 12)))
           for r in (("pm", "cls") if os.environ.get("KT_CLS", "1") == "1" else ("pm",))]
if os.environ.get("KT_POS") == "gblocks":
    LAYERS = GBLOCKS
if os.environ.get("KT_POS") == "interv":
    LAYERS = INTERV
# KT_ARCH=vit: the same three slots map to ViT-B/16 blocks 4 / 8 / 12 (mean of patch tokens)
ARCH = os.environ.get("KT_ARCH", "resnet")
VIT_BLOCKS = {"layer2": 3, "layer3": 7, "layer4": 11}
DIMS = {} if ARCH == "gen" else {l: 768 for l in LAYERS} if ARCH == "vit" else {"layer2": 512, "layer3": 1024, "layer4": 2048,
                                                       "l3": 1024, "l4b0": 2048, "l4b1": 2048, "l4b2": 2048}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Net(nn.Module):
    """ResNet-50 taking [0,1] pixels; returns logits and GAP features of LAYERS."""

    def __init__(self, base):
        super().__init__()
        self.m = base
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        m = self.m
        x = (x - self.mean) / self.std
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer1(x)
        if LAYERS is RDISSECT:
            f3 = m.layer3(m.layer2(x))
            b0 = m.layer4[0](f3)
            b1 = m.layer4[1](b0)
            b2 = m.layer4[2](b1)
            feats = {n: t.mean((2, 3)) for n, t in [("l3", f3), ("l4b0", b0), ("l4b1", b1), ("l4b2", b2)]}
            return m.fc(torch.flatten(m.avgpool(b2), 1)), feats
        f2 = m.layer2(x)
        f3 = m.layer3(f2)
        f4 = m.layer4(f3)
        feats = {"layer2": f2.mean((2, 3)), "layer3": f3.mean((2, 3)), "layer4": f4.mean((2, 3))}
        logits = m.fc(torch.flatten(m.avgpool(f4), 1))
        return logits, feats


CLEAN = None  # clean inputs of the current search batch (set by run_model) for the frozen-attention modes


def set_clean(x):
    global CLEAN
    CLEAN = x


def _clean_for(x):
    return CLEAN if CLEAN is not None and CLEAN.shape == x.shape else x.detach()


def _qkv(attn, h, tv):
    """q, k, v [B, H, N, D] of a self-attention module (tv: torchvision nn.MultiheadAttention, else timm Attention)."""
    B, N, E = h.shape
    H = attn.num_heads
    qkv = nn.functional.linear(h, attn.in_proj_weight, attn.in_proj_bias) if tv else attn.qkv(h)
    q, k, v = [t.view(B, N, H, E // H).transpose(1, 2) for t in qkv.chunk(3, -1)]
    if not tv:
        q, k = attn.q_norm(q), attn.k_norm(k)
    return q, k, v


def _probs(q, k, T=1.0):
    return torch.softmax(q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5) / T, -1)


def _attn_edit(attn, h, mode, tv, h0=None):
    """Self-attention with an edit. mode: none | uniform | h<k> (head k zeroed) | t<T> (softmax temperature T) |
    frozenA = A(x0)V(x) | frozenV = A(x)V(x0) | frozenAV = A(x0)V(x0), where x0 is the clean input (h0 = its LN'd tokens)."""
    q, k, v = _qkv(attn, h, tv)
    N = h.shape[1]
    if mode == "uniform":
        o = v.mean(2, keepdim=True).expand(-1, -1, N, -1)
    elif mode.startswith("t"):
        o = _probs(q, k, float(mode[1:])) @ v
    elif mode.startswith("frozen"):
        with torch.no_grad():
            q0, k0, v0 = _qkv(attn, h0, tv)
            A0 = _probs(q0, k0)
        A = A0 if mode in ("frozenA", "frozenAV") else _probs(q, k)
        o = A @ (v0 if mode in ("frozenV", "frozenAV") else v)
    else:
        o = nn.functional.scaled_dot_product_attention(q, k, v)
    if mode.startswith("h"):
        keep = torch.ones(o.shape[1], device=h.device, dtype=h.dtype)
        keep[int(mode[1:])] = 0
        o = o * keep.view(1, -1, 1, 1)
    o = o.transpose(1, 2).reshape(h.shape)
    return attn.out_proj(o) if tv else attn.proj(getattr(attn, "norm", nn.Identity())(o))


def _mha(attn, h, mode):
    return _attn_edit(attn, h, mode, True)


class ViTNet(nn.Module):
    """ViT-B/16 taking [0,1] pixels; returns logits and patch-token means after VIT_BLOCKS."""

    def __init__(self, base):
        super().__init__()
        self.m = base
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())
        self.at = {v: k for k, v in VIT_BLOCKS.items()}

    def embed(self, x):
        m = self.m
        x = m._process_input((x - self.mean) / self.std)
        x = torch.cat([m.class_token.expand(x.shape[0], -1, -1), x], 1)
        return m.encoder.dropout(x + m.encoder.pos_embedding)

    def forward(self, x):
        m = self.m
        enc = m.encoder
        xin, x = x, self.embed(x)
        feats = {}
        if LAYERS is INTERV:
            x0 = None
            if INTERV_MODE.startswith("frozen"):
                with torch.no_grad():
                    x0 = self.embed(_clean_for(xin))
            for i, blk in enumerate(enc.layers):
                mode = INTERV_MODE if i == INTERV_BLK else "none"
                if mode == "noattn":
                    p_attn = x
                else:
                    h0 = blk.ln_1(x0) if x0 is not None and i == INTERV_BLK else None
                    p_attn = x + blk.dropout(_attn_edit(blk.self_attention, blk.ln_1(x), mode, True, h0))
                x = p_attn + blk.mlp(blk.ln_2(p_attn))
                if x0 is not None and i < INTERV_BLK:
                    with torch.no_grad():
                        x0 = blk(x0)
            p_ln = enc.ln(x)
            feats["mlp_pm"], feats["ln_cls"] = x[:, 1:].mean(1), p_ln[:, 0]
            return m.heads(p_ln[:, 0]), feats
        if LAYERS is BLOCKS:
            for i, blk in enumerate(enc.layers):
                x = blk(x)
                feats[f"k{i}_pm"], feats[f"k{i}_cls"] = x[:, 1:].mean(1), x[:, 0]
            return m.heads(enc.ln(x)[:, 0]), feats
        if LAYERS is DISSECT:
            for blk in enc.layers[:-1]:
                x = blk(x)
            blk = enc.layers[-1]
            h = blk.ln_1(x)
            h, _ = blk.self_attention(h, h, h, need_weights=False)
            p_attn = x + blk.dropout(h)
            p_mlp = p_attn + blk.mlp(blk.ln_2(p_attn))
            p_ln = enc.ln(p_mlp)
            for name, t in [("b11", x), ("attn", p_attn), ("mlp", p_mlp), ("ln", p_ln)]:
                feats[name + "_pm"] = t[:, 1:].mean(1)
                feats[name + "_cls"] = t[:, 0]
            return m.heads(p_ln[:, 0]), feats
        for i, blk in enumerate(enc.layers):
            x = blk(x)
            if i in self.at:
                feats[self.at[i]] = x[:, 1:].mean(1)
        logits = m.heads(enc.ln(x)[:, 0])
        return logits, feats


TIMM = {"deit": "deit_base_patch16_224.fb_in1k", "augreg": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "deit3": "deit3_base_patch16_224.fb_in1k",
        # same wrapper (dissect / blocks / interv) for the other standard timm ViTs
        "deit_s": "deit_small_patch16_224.fb_in1k", "deit3_s": "deit3_small_patch16_224.fb_in1k",
        "augreg_s": "vit_small_patch16_224.augreg_in21k_ft_in1k", "deit_dist": "deit_base_distilled_patch16_224.fb_in1k",
        "clip_ft": "vit_base_patch16_clip_224.openai_ft_in1k"}


class TimmViTNet(nn.Module):
    """timm ViT-B/16 taking [0,1] pixels; KT_POS=dissect positions inside the last block (same names as ViTNet)."""

    def __init__(self, base):
        super().__init__()
        self.m = base
        c = base.pretrained_cfg
        self.register_buffer("mean", torch.tensor(c["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(c["std"]).view(1, 3, 1, 1))

    def embed(self, x):
        m = self.m
        return m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed((x - self.mean) / self.std))))

    def forward(self, x):
        m = self.m
        xin, x = x, self.embed(x)
        k = m.num_prefix_tokens
        feats = {}
        if LAYERS is BLOCKS:
            for i, blk in enumerate(m.blocks):
                x = blk(x)
                feats[f"k{i}_pm"], feats[f"k{i}_cls"] = x[:, k:].mean(1), x[:, 0]
            return m.forward_head(m.norm(x)), feats
        if LAYERS is INTERV:
            x0 = None
            if INTERV_MODE.startswith("frozen"):
                with torch.no_grad():
                    x0 = self.embed(_clean_for(xin))
            for i, blk in enumerate(m.blocks):
                if i == INTERV_BLK and INTERV_MODE != "none":
                    if INTERV_MODE != "noattn":
                        h0 = blk.norm1(x0) if x0 is not None else None
                        x = x + blk.drop_path1(blk.ls1(_attn_edit(blk.attn, blk.norm1(x), INTERV_MODE, False, h0)))
                    x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
                else:
                    x = blk(x)
                if x0 is not None and i < INTERV_BLK:
                    with torch.no_grad():
                        x0 = blk(x0)
            p_ln = m.norm(x)
            feats["mlp_pm"], feats["ln_cls"] = x[:, k:].mean(1), p_ln[:, 0]
            return m.forward_head(p_ln), feats
        assert LAYERS is DISSECT, "timm models are wired for KT_POS=dissect / blocks / interv"
        for blk in m.blocks[:-1]:
            x = blk(x)
        blk = m.blocks[-1]
        p_attn = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        p_mlp = p_attn + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(p_attn))))
        p_ln = m.norm(p_mlp)
        for name, t in [("b11", x), ("attn", p_attn), ("mlp", p_mlp), ("ln", p_ln)]:
            feats[name + "_pm"] = t[:, k:].mean(1)
            feats[name + "_cls"] = t[:, 0]
        return m.forward_head(p_ln), feats


GEN = {"deit_s": "deit_small_patch16_224.fb_in1k", "deit3_s": "deit3_small_patch16_224.fb_in1k",
       "augreg_s": "vit_small_patch16_224.augreg_in21k_ft_in1k", "augreg_t": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
       "augreg_l": "vit_large_patch16_224.augreg_in21k_ft_in1k", "deit_dist": "deit_base_distilled_patch16_224.fb_in1k",
       "beit": "beit_base_patch16_224.in22k_ft_in22k_in1k", "clip_ft": "vit_base_patch16_clip_224.openai_ft_in1k",
       "mixer": "mixer_b16_224.goog_in21k_ft_in1k", "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",
       "convnext_t": "convnext_tiny.fb_in1k",
       # same ResNet-50 architecture, three training recipes (torchvision v1 / torchvision v2 / timm A1)
       "rn50_tv": "resnet50.tv_in1k", "rn50_tv2": "resnet50.tv2_in1k", "rn50_a1": "resnet50.a1_in1k",
       "deit_g": "deit_base_patch16_224.fb_in1k", "augreg_g": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
       "deit3_g": "deit3_base_patch16_224.fb_in1k",
       # depth-only pair (same width 192, same recipe): depth branch Phase 0
       "cait_xxs24": "cait_xxs24_224.fb_dist_in1k", "cait_xxs36": "cait_xxs36_224.fb_dist_in1k",
       # stage-3 depth-only pairs with convnext_t / swin_t (same widths, same recipe): depth Phase 0 replication
       "convnext_s": "convnext_small.fb_in1k", "swin_s": "swin_small_patch4_window7_224.ms_in1k"}


class GenericNet(nn.Module):
    """Any timm classifier taking [0,1] pixels; features = outputs of every block (forward hooks), pooled over
    tokens / space (pm) plus the first token (cls) for models with prefix tokens. Names follow GBLOCKS."""

    def __init__(self, base, name):
        super().__init__()
        self.m = base
        c = base.pretrained_cfg
        self.register_buffer("mean", torch.tensor(c["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(c["std"]).view(1, 3, 1, 1))
        self.kind = "nhwc" if "swin" in name else "nchw" if ("convnext" in name or "rn50" in name) else "tok"
        if self.kind == "nhwc":
            blocks = [b for l in base.layers for b in l.blocks]
        elif self.kind == "nchw" and hasattr(base, "stages"):
            blocks = [b for st in base.stages for b in st.blocks]
        elif self.kind == "nchw":  # ResNet: the 16 bottlenecks of layer1-4
            blocks = [b for l in (base.layer1, base.layer2, base.layer3, base.layer4) for b in l]
        else:
            blocks = list(base.blocks)
        self.k = (getattr(base, "num_prefix_tokens", 0) or 0) if self.kind == "tok" else 0  # CaiT: None, no CLS in patch blocks
        self.nblk = len(blocks)
        self.blocks = blocks
        self._f = {}
        # residual-scaling intervention : block KT_ABLK outputs x + KT_ALPHA * (block(x) - x).
        # Registered before the capture hooks so that the captured output is the scaled one.
        ablk, alpha = int(os.environ.get("KT_ABLK", -1)), float(os.environ.get("KT_ALPHA", 1.0))
        if ablk >= 0:
            blocks[ablk].register_forward_hook(lambda mod, inp, out: inp[0] + alpha * (out - inp[0]))
        for i, b in enumerate(blocks):
            b.register_forward_hook(lambda mod, inp, out, i=i: self._f.__setitem__(i, out))

    def forward(self, x):
        self._f.clear()
        lg = self.m((x - self.mean) / self.std)
        feats = {}
        for i, t in self._f.items():
            if self.kind == "tok":
                feats[f"k{i}_pm"] = t[:, self.k:].mean(1)
                if self.k:
                    feats[f"k{i}_cls"] = t[:, 0]
            else:
                feats[f"k{i}_pm"] = t.mean((1, 2)) if self.kind == "nhwc" else t.mean((2, 3))
        return lg, {l: feats[l] for l in LAYERS}


def load_model(name):
    if name in GEN and ARCH == "gen":
        import timm
        import huggingface_hub
        from safetensors.torch import load_file
        base = timm.create_model(GEN[name], pretrained=False)
        sd = load_file(huggingface_hub.hf_hub_download("timm/" + GEN[name], "model.safetensors"))
        own = base.state_dict()
        # checkpoints of BEiT / Swin also store derived buffers (relative_position_index, attn_mask) that this timm
        # version rebuilds itself; drop only those, everything else must match
        extra = [k for k in sd if k not in own]
        assert all(k.endswith(("relative_position_index", "attn_mask")) for k in extra), extra[:5]
        missing = base.load_state_dict({k: v for k, v in sd.items() if k in own}, strict=False).missing_keys
        assert not [k for k in missing if not k.endswith(("relative_position_index", "attn_mask"))], missing[:5]
        net = GenericNet(base, name).eval()
        assert net.nblk == int(os.environ.get("KT_NBLK", 12)) and (net.k > 0) == (os.environ.get("KT_CLS", "1") == "1"), \
            f"{name}: {net.nblk} blocks, {net.k} prefix tokens"
        with torch.no_grad():
            _, fs = net(torch.rand(1, 3, 224, 224))
        DIMS.update({l: f.shape[1] for l, f in fs.items()})
        return net
    if name in TIMM:
        import timm
        import huggingface_hub
        from safetensors.torch import load_file
        base = timm.create_model(TIMM[name], pretrained=False)
        base.load_state_dict(load_file(huggingface_hub.hf_hub_download("timm/" + TIMM[name], "model.safetensors")))
        net = TimmViTNet(base).eval()
        with torch.no_grad():
            _, fs = net(torch.rand(1, 3, 224, 224))
        DIMS.update({l: f.shape[1] for l, f in fs.items()})  # ViT-S has 384-d tokens
        return net
    if name in ("vit", "vit_random"):
        assert ARCH == "vit", "set KT_ARCH=vit"
        torch.manual_seed(0)
        base = torchvision.models.vit_b_16()
        if name == "vit":
            base.load_state_dict(torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth"),
                                            map_location="cpu"))
        else:  # torchvision zero-inits the head, which would make every logit 0
            nn.init.normal_(base.heads.head.weight, std=768 ** -0.5)
        return ViTNet(base).eval()
    base = torchvision.models.resnet50()
    if name == "std":
        sd = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"), map_location="cpu")
        base.load_state_dict(sd)
    elif name == "robust":
        ck = torch.load(os.path.join(DATA, "resnet50_l2_eps3.ckpt"), map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        sd = {k[len("module.model."):]: v for k, v in sd.items() if k.startswith("module.model.")}
        base.load_state_dict(sd)
    elif name == "random":
        torch.manual_seed(0)
        base = torchvision.models.resnet50()
    else:
        raise ValueError(name)
    return Net(base).eval()


def class_index():
    with open(os.path.join(DATA, "imagenet_class_index.json")) as f:
        d = json.load(f)
    return {v[0]: int(k) for k, v in d.items()}


SMOKE = int(os.environ.get("SMOKE", "0"))


def load_cache(name):
    d = np.load(os.path.join(CACHE, f"{name}.npz"))
    x, y = d["x"], d["y"]
    paths.verify(name, x, y)
    if SMOKE:
        n = 2 if name == "id_eval" else (max(SMOKE, 40) if name == "id_fit" else SMOKE)
        x, y = x[:n], y[:n]
    return x, y


def to_float(x_uint8, device):
    return torch.from_numpy(x_uint8).to(device).float().div_(255.0)
