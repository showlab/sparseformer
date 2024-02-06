import torch
import re

CONVERT_V1_TO_V2_RULES = [
    (r'tuo_proj', 'out_proj'),
    (r'token_roi\.weight', 'initial_roi.weight'),
    (r'token_embedding\.weight', 'initial_embedding.weight'),
    (r'ffn.0.', 'norm2.'),
    (r'ln_attn.', 'norm1.'),
    (r'ffn.1.', 'mlp.fc1.'),
    (r'ffn.3.', 'mlp.fc2.'),
    (r'attn.in_proj_', 'attn.qkv.'),
    (r'attn.out_proj.', 'attn.proj.'),
]

def convert_ckpt_v1_to_v2(state_dict: dict):
    converted_dict = dict()
    keys = list(state_dict.keys())

    for key in keys:
        new_key = key
        for pattern, replacement in CONVERT_V1_TO_V2_RULES:
            prev_key = new_key
            new_key = re.sub(pattern, replacement, prev_key)
            if new_key != prev_key: # only match once
                break
        converted_dict[new_key] = state_dict[key]

    return converted_dict