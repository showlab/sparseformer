from .modelling import OP
from .utils import CompatibleAttrDict

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
    )
    
def make_btsp_attr():
    return CompatibleAttrDict(
        pg_inner_dim=128,
        type_num_attn_heads=2,
        type_op_adjust=2,
        mixing_bias=False,
        transition_ln=False,
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