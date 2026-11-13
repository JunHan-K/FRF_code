import timm, huggingface_hub as h
from safetensors.torch import load_file
for n in ['vit_base_patch16_224.augreg2_in21k_ft_in1k', 'deit3_base_patch16_224.fb_in1k']:
    f = h.hf_hub_download('timm/' + n, 'model.safetensors')
    m = timm.create_model(n, pretrained=False); print(m.load_state_dict(load_file(f))); c = m.pretrained_cfg
    print(n, f, c['mean'], c['std'], c.get('crop_pct'), m.global_pool, type(m.blocks[-1].ls1).__name__, m.num_prefix_tokens, type(m.fc_norm).__name__, flush=True)
