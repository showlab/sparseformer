import torch
import re

DELETE = "DELETE"

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
    (r'ln_post.', 'head.norm.'),
    (r'global_embedding', 'cls_embedding'),
    (r'cls_head', 'head.classifier'),
    (r'global_roi', DELETE),
    (r'stage_embedding', DELETE),
]


def convert_ckpt_v1_to_v2(state_dict: dict) -> dict:
    converted_dict = dict()
    keys = list(state_dict.keys())

    for key in keys:
        new_key = key
        for pattern, replacement in CONVERT_V1_TO_V2_RULES:
            prev_key = new_key
            new_key = re.sub(pattern, replacement, prev_key)
            if new_key != prev_key: # only match once
                break
        if DELETE in new_key:
            continue
        converted_dict[new_key] = state_dict[key]

    return converted_dict

def convert_ckpt_btsp_to_v2(state_dict: dict) -> dict:
    state_dict = convert_ckpt_v1_to_v2(state_dict)


    converted_dict = dict()

    roi_cc = state_dict.pop("token_roi_cc.weight")
    roi_wh = state_dict.pop("token_roi_wh.weight")
    converted_dict["initial_roi.weight"] = torch.cat([roi_cc - 0.5*roi_wh, roi_cc + 0.5*roi_wh], dim=1)

    keys = list(state_dict.keys())
    max_layer = 0
    for key in keys:
        if key.startswith('layers'):
            matched = re.search(r"layers\.(\d+)\.", key)
            if matched:
                max_layer = max(max_layer, int(matched.group(1)))


    for key in keys:
        new_key = key
        if key.startswith('blocks'):
            matched = re.search(r"^blocks\.(\d+)\.", key)
            if matched:
                block_idx = int(matched.group(1))
            new_key = re.sub(r"^blocks\.(\d+)\.", f"layers.{max_layer+block_idx+1}.", key)
        converted_dict[new_key] = state_dict[key]

    return converted_dict