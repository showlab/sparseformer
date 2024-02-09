import math

import torch
import torch.nn as nn

from torch import Tensor


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class CompatibleAttrDict(AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_num_attn_heads = 2
        self.type_op_adjust = 2
        self.restrict_grad_norm = False
        self.pg_inner_dim = .25
        self.transition_ln = False
        self.mixing_bias = False
        self.ln_eps = 1e-5
        self.use_stage_embedding = True
        self.clip_quick_gelu = False
        self.grid_sample_config = dict(mode="bilinear", padding_mode="border", align_corners=False)
        self.update(**kwargs)

def _maybe_promote(x: Tensor) -> Tensor:
    """
    Credits to Meta's xformers (xformers/components/attention/favor.py)
    """
    # Only promote fp16 buffers, bfloat16 would be fine for instance
    return x.float() if x.dtype == torch.float16 else x



@torch.no_grad()
def init_layer_norm_unit_norm(layer: nn.LayerNorm, gamma=1.0):
    assert len(layer.normalized_shape) == 1
    width = layer.normalized_shape[0]
    nn.init.ones_(layer.weight)
    layer.weight.data.mul_(gamma * (width**-0.5))


@torch.no_grad()
def init_linear_params(weight, bias):
    std = 0.02
    nn.init.trunc_normal_(weight, std=std)
    if bias is not None:
        nn.init.constant_(bias, 0)


class RestrictGradNorm(torch.autograd.Function):
    GRAD_SCALE = 1.0  # variable tracking grad scale in amp

    @staticmethod
    def forward(ctx, x, norm=0.1):
        ctx.norm = norm
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # amp training scales up the grad scale for the entire network
        norm = ctx.norm * RestrictGradNorm.GRAD_SCALE
        grad_x = grad_output.clone().clamp(-norm, norm)
        return grad_x, None


def drop_path(x, drop_prob: float = 0.0, training: bool = False, batch_first=False):
    """
    Copyright 2020 Ross Wightman
    """
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    if batch_first:
        shape = (x.shape[1], 1) + (1,) * (x.ndim - 2)
    else:
        shape = (1, x.shape[1],) + (1,) * (x.ndim - 2)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Copyright 2020 Ross Wightman
    """

    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, batch_first=False):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_first = batch_first

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.batch_first)
