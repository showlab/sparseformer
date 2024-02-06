from .modelling import OP
from .utils import CompatibleAttrDict

def make_v1_ops_list(num_cortex):
    return \
    [[OP.ATTN, OP.ADJ, OP.SAM, OP.MLP]] + \
    [[OP.ATTN, OP.ADJ, OP.SAM, OP.MLP]] + \
    [[OP.ATTN, OP.MLP]] * (num_cortex - 1)

def make_btsp_ops_list(num_cortex):
    return \
    [[OP.SAM, OP.PE_INJ, OP.ATTN, OP.MLP, OP.ADJ]] + \
    [[OP.SAM, OP.PE_INJ]] + \
    [[OP.ATTN, OP.MLP]] * num_cortex

def make_btsp_attr():
    return CompatibleAttrDict(
        pg_inner_dim=128,
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
