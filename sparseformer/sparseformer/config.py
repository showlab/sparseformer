from .modeling import OP
from .utils import CompatibleAttrDict

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, \
                                IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
                                OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

def route_name_to_config(model_type: str):
    model_type = model_type.lower()
    global_dict = globals()
    if model_type in global_dict:
        config = global_dict[model_type]()
        config.update(model_type=model_type)
        return config
    assert False


def make_v1_ops_list(num_cortex):
    return \
    [(OP.ATTN, OP.ROI_ADJ, OP.SAMPLING_M, OP.MLP)] + \
    [(OP.ATTN, OP.ROI_ADJ, OP.SAMPLING_M, OP.MLP)] + \
    [(OP.ATTN, OP.MLP)] * (num_cortex-1)


def make_btsp_ops_list(num_cortex):
    return \
    [(OP.SAMPLING_M, OP.PE_INJECT, OP.ATTN, OP.MLP, OP.ROI_ADJ)] + \
    [(OP.SAMPLING_M, OP.PE_INJECT,)] + \
    [(OP.ATTN, OP.MLP)] * (num_cortex)


def make_v1_attr():
    return CompatibleAttrDict(
        type_num_attn_heads=1,
        type_op_adjust=1,
        mixing_bias=True,
        transition_ln=True,
        ln_eps=1e-6,
        use_stage_embedding=False,
        grid_sample_config=dict(mode="bilinear", padding_mode="zeros", align_corners=False)
    )

    
def make_btsp_attr():
    return CompatibleAttrDict(
        pg_inner_dim=128,
        grid_sample_config=dict(mode="bilinear", padding_mode="border", align_corners=False)
    )


def make_clip_attr():
    attr = make_btsp_attr()
    attr.update(clip_quick_gelu=True)
    return attr


def base_v1_config():
    return dict(
        conv_dim=96,
        num_latent_tokens=49,
        num_sampling_points=36,
        same_token_embedding=False,
        width_configs=[256,] + [512,] * 8,
        repeats=[4,] + [1,] * 8,
        head_op="avg",
        ops_list=make_v1_ops_list(8),
        compatible_config=make_v1_attr(),
    )


def base_btsp_config():
    return dict(
        conv_dim=64,
        num_latent_tokens=48,
        num_sampling_points=16,
        same_token_embedding=True,
        cls_token_insert_layer=2,
        width_configs=[384, 768, ] + [768,] * 8,
        repeats=[4, 1,] + [1,] * 8,
        head_op="cls",
        ops_list=make_btsp_ops_list(8),
        compatible_config=make_btsp_attr(),
    )


def sparseformer_v1_tiny(num_latent_tokens=49):
    config = base_v1_config()
    config.update(
        num_latent_tokens=num_latent_tokens,
        width_configs=[256,] + [512,] * 8,
        repeats=[4,] + [1,] * 8,
        ops_list=make_v1_ops_list(8),
        compatible_config=make_v1_attr(),
        proj_dim=1000,
        proj_bias=True,
        drop_path_rate=0.2,
        normalization_factor=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    )
    return config


def sparseformer_v1_small(num_latent_tokens=64):
    config = base_v1_config()
    config.update(
        num_latent_tokens=num_latent_tokens,
        width_configs=[320,] + [640,] * 8,
        repeats=[4,] + [1,] * 8,
        ops_list=make_v1_ops_list(8),
        compatible_config=make_v1_attr(),
        proj_dim=1000,
        proj_bias=True,
        drop_path_rate=0.3,
        normalization_factor=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    )
    return config


def sparseformer_v1_base(num_latent_tokens=81):
    config = base_v1_config()
    config.update(
        num_latent_tokens=num_latent_tokens,
        width_configs=[384,] + [768,] * 10,
        repeats=[4,] + [1,] * 10,
        ops_list=make_v1_ops_list(10),
        compatible_config=make_v1_attr(),
        proj_dim=1000,
        proj_bias=True,
        drop_path_rate=0.4,
        normalization_factor=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    )
    return config
    

def sparseformer_btsp_augreg_base(num_latent_tokens=49):
    config = base_btsp_config()
    config.update(
        num_latent_tokens=num_latent_tokens-1,
        width_configs=[384, 768,] + [768,] * 8,
        repeats=[4, 1,] + [1,] * 8,
        ops_list=make_btsp_ops_list(8),
        compatible_config=make_btsp_attr(),
        proj_dim=1000,
        proj_bias=True,
        normalization_factor=(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
    )
    return config


def sparseformer_btsp_augreg_large(num_latent_tokens=49):
    config = base_btsp_config()
    config.update(
        num_latent_tokens=num_latent_tokens-1,
        width_configs=[512, 1024,] + [1024,] * 16,
        repeats=[4, 1,] + [1,] * 16,
        ops_list=make_btsp_ops_list(16),
        compatible_config=make_btsp_attr(),
        proj_dim=1000,
        proj_bias=True,
        normalization_factor=(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
    )
    return config


def sparseformer_btsp_openai_clip_base(num_latent_tokens=49):
    config = base_btsp_config()
    config.update(
        num_latent_tokens=num_latent_tokens-1,
        width_configs=[384, 768,] + [768,] * 8,
        repeats=[4, 1,] + [1,] * 8,
        ops_list=make_btsp_ops_list(8),
        compatible_config=make_clip_attr(),
        proj_dim=512,
        proj_bias=False,
        normalization_factor=(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    )
    return config


def sparseformer_btsp_openai_clip_large(num_latent_tokens=64):
    config = base_btsp_config()
    config.update(
        num_latent_tokens=num_latent_tokens-1,
        width_configs=[512, 1024,] + [1024,] * 16,
        repeats=[4, 1,] + [1,] * 16,
        ops_list=make_btsp_ops_list(16),
        compatible_config=make_clip_attr(),
        proj_dim=768,
        proj_bias=False,
        normalization_factor=(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    )
    return config
