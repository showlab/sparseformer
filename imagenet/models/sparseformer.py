# --------------------------------------------------------
# SparseFormer
# Copyright 2023 Ziteng Gao
# Licensed under The MIT License
# Written by Ziteng Gao
# --------------------------------------------------------

from functools import partial
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


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
    layer.weight.data.mul_(gamma * (width ** -0.5))


@torch.no_grad()
def init_linear_params(weight, bias):
    std = .02
    nn.init.trunc_normal_(weight, std=std)
    if bias is not None:
        nn.init.constant_(bias, 0)


class RestrictGradNorm(torch.autograd.Function):
    GRAD_SCALE = 1.0 # variable tracking grad scale in amp
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


def drop_path(x, drop_prob: float = 0., training: bool = False):
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

    ** IMPORTANT **
    Modified (Jun. 16, by Ziteng Gao):
    since we use the second dimension as the batch dimension, the random tensor shape is
    actually `(1, x.shape[1],) + (1,) * (x.ndim - 2)` (not the originally `(x.shape[0],)
    +(1,) * (x.ndim - 2)` in timm).
    Sorry for this bug since I simply adopted timm code in the code reorganization
    without further investigation.
    ** This corrected version aligns with the paper **
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (1, x.shape[1],) + (1,) * (x.ndim - 2)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
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

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@autocast(enabled=False)
def roi_adjust(token_roi: Tensor, token_adjust: Tensor):
    token_xy = (token_roi[..., 2:]+token_roi[..., :2]) * 0.5
    token_wh = (token_roi[..., 2:]-token_roi[..., :2]).abs()

    token_xy = token_xy + token_adjust[..., :2]*token_wh
    token_wh = token_wh * token_adjust[..., 2:].exp()

    token_roi_new = torch.cat(
        [token_xy-0.5*token_wh, token_xy+0.5*token_wh], dim=-1)
    return token_roi_new


@autocast(enabled=False)
def make_absolute_sampling_point(
        token_roi: Tensor,
        sampling_offset: Tensor,
        num_heads: int,
        num_points: int) -> Tensor:
    batch, num_tokens, _ = sampling_offset.shape

    sampling_offset = sampling_offset.view(batch, num_tokens, 1, num_heads*num_points, 2)

    offset_mean = sampling_offset.mean(-2, keepdim=True)
    offset_std = sampling_offset.std(-2, keepdim=True)+1e-7
    sampling_offset = (sampling_offset - offset_mean)/(3*offset_std)

    token_roi_xy = (token_roi[:, :, 2:]+token_roi[:, :, :2])/2.0
    token_roi_wh = token_roi[:, :, 2:]-token_roi[:, :, :2]
    sampling_offset = sampling_offset[..., :2] * token_roi_wh.view(batch, num_tokens, 1, 1, 2)
    sampling_absolute = token_roi_xy.view(batch, num_tokens, 1, 1, 2) + sampling_offset

    return sampling_absolute


@autocast(enabled=False)
def sampling_from_img_feat(
        sampling_point: Tensor,
        img_feat: Tensor,
        n_points=1):
    batch, Hq, Wq, num_points_per_head, _ = sampling_point.shape
    batch, channel, height, width = img_feat.shape

    n_heads = num_points_per_head//n_points

    sampling_point = sampling_point.view(batch, Hq, Wq, n_heads, n_points, 2) \
        .permute(0, 3, 1, 2, 4, 5).contiguous().flatten(0, 1)
    # We truncate the grad for sampling coordinates to the unit length (e.g., 1.0/height)
    # to avoid inaccurate gradients due to bilinear sampling. That is, we restrict
    # gradients to be local.
    sampling_point = RestrictGradNorm.apply(sampling_point, 1.0/height)
    sampling_point = sampling_point.flatten(2, 3)
    sampling_point = sampling_point*2.0-1.0
    img_feat = img_feat.view(batch*n_heads, channel//n_heads, height, width)
    out = F.grid_sample(
        img_feat, sampling_point,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )

    out = out.view(batch, n_heads, channel//n_heads, Hq, Wq, n_points)

    return out.permute(0, 3, 4, 1, 5, 2).flatten(1, 2)


@autocast(enabled=False)
def layer_norm_by_dim(x: Tensor, dim=-1):
    mean = x.mean(dim, keepdim=True)
    x = x - mean
    std = (x.var(dim=dim, keepdim=True)+1e-7).sqrt()
    return x / std


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, out_query_dim=None):
        super(AdaptiveMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        out_query_dim = out_query_dim if out_query_dim is not None else query_dim

        self.query_dim = query_dim
        self.out_query_dim = out_query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim//n_groups
        self.eff_out_dim = self.eff_in_dim

        self.channel_param_count = (self.eff_in_dim * self.eff_out_dim)
        self.spatial_param_count = (self.in_points * self.out_points)

        self.total_param_count = self.channel_param_count + self.spatial_param_count

        self.parameter_generator = nn.Sequential(
            LAYER_NORM(self.query_dim),
            nn.Linear(self.query_dim, self.query_dim//4),
            nn.Linear(self.query_dim//4, self.total_param_count*self.n_groups),
        )

        self.m_beta = nn.Parameter(torch.zeros(self.eff_out_dim))
        self.s_beta = nn.Parameter(torch.zeros(self.out_points))

        # the name should be `out_proj` but ...
        self.out_proj = nn.Linear(n_groups*self.eff_out_dim*self.out_points,
                                  self.out_query_dim)

        self.act = nn.GELU()

    @torch.no_grad()
    def init_layer(self):
        init_layer_norm_unit_norm(self.parameter_generator[0], gamma=1.)
        nn.init.zeros_(self.parameter_generator[-1].weight)

        bias = self.parameter_generator[-1].bias
        nn.init.xavier_uniform_(
            bias[:self.eff_in_dim*self.eff_out_dim].view(
                self.eff_in_dim, self.eff_out_dim), gain=1
        )
        nn.init.xavier_uniform_(
            bias[self.eff_in_dim*self.eff_out_dim:].view(
                self.in_points, self.out_points), gain=1
        )

    def forward(self, sampled: Tensor, query: Tensor):
        batch, num_tokens, g, num_points, num_channel = sampled.size()
        assert g == 1
        out = sampled.reshape(batch*num_tokens, num_points, num_channel)

        params = self.parameter_generator(query)
        params = params.reshape(batch*num_tokens, -1)

        channel_mixing, spatial_mixing = params.split_with_sizes(
            [self.eff_in_dim*self.eff_out_dim, self.out_points*self.in_points],
            dim=-1
        )

        channel_mixing = channel_mixing.reshape(batch*num_tokens, self.eff_in_dim, self.eff_out_dim)
        spatial_mixing = spatial_mixing.reshape(batch*num_tokens, self.out_points, self.in_points)

        channel_bias = self.m_beta.view(1, 1, self.eff_out_dim)
        spatial_bias = self.s_beta.view(1, self.out_points, 1)

        out = torch.baddbmm(channel_bias, out, channel_mixing)
        out = self.act(out)

        out = torch.baddbmm(spatial_bias, spatial_mixing, out)
        out = self.act(out)

        out = out.reshape(batch, num_tokens, -1)
        out = self.out_proj(out)

        return out


class SFUnit(nn.Module):
    def __init__(self,
                 dim,
                 conv_dim,
                 num_sampling_points,
                 sampling_enabled=False,
                 adjusting_enabled=False,
                 final_adjusting=False,
                 mlp_ratio=4,
                 repeats=1,
                 drop_path=0.0):
        super(SFUnit, self).__init__()
        self.dim = dim
        self.conv_dim = dim

        self.num_sampling_points = num_sampling_points

        self.sampling_enabled = sampling_enabled
        self.adjusting_enabled = adjusting_enabled
        self.final_adjusting = final_adjusting
        self.repeats = repeats

        if self.sampling_enabled:
            self.adaptive_mixing = AdaptiveMixing(
                conv_dim, num_sampling_points, 1, query_dim=dim, out_query_dim=dim
            )
        else:
            self.adaptive_mixing = None

        self.ffn = nn.Sequential(
            LAYER_NORM(dim),
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim),
        )

        self.attn = nn.MultiheadAttention(
            dim,
            64 if dim >= 64 else dim,
            dropout=0.0
        )

        self.ln_attn = LAYER_NORM(dim)

        if self.sampling_enabled:
            self.roi_offset_module = nn.Sequential(
                LAYER_NORM(dim),
                nn.Linear(dim, num_sampling_points*2, bias=False),
            )
            self.roi_offset_bias = nn.Parameter(
                torch.randn(num_sampling_points*2)
            )

        if self.adjusting_enabled or self.final_adjusting:
            self.roi_adjust_module = nn.Sequential(
                LAYER_NORM(dim),
                nn.Linear(dim, 4, bias=False),
            )
        self.drop_path = drop_path
        self.dropout = DropPath(
            drop_path if isinstance(drop_path, (float, int)) else .0
        )

    @torch.no_grad()
    def init_layer(self):
        init_linear_params(self.attn.in_proj_weight, self.attn.in_proj_bias)

        if self.adjusting_enabled:
            init_layer_norm_unit_norm(self.roi_adjust_module[0],  1.)
            nn.init.zeros_(self.roi_adjust_module[-1].weight)
            pass

        if self.sampling_enabled:
            init_layer_norm_unit_norm(self.roi_offset_module[0],  1.)
            nn.init.zeros_(self.roi_adjust_module[-1].weight)

            root = int(self.num_sampling_points**0.5)
            x = torch.linspace(0.5/root, 1-0.5/root, root)\
                .view(1, -1, 1).repeat(root, 1, 1)
            y = torch.linspace(0.5/root, 1-0.5/root, root)\
                .view(-1, 1, 1).repeat(1, root, 1)
            grid = torch.cat([x, y], dim=-1).view(root**2, -1)
            bias = self.roi_offset_bias.view(-1, 2)
            bias.data[:root**2] = grid - 0.5

    def shift_token_roi(self, token_embedding: Tensor, token_roi: Tensor):
        roi_adjust_logit = self.roi_adjust_module(token_embedding)
        roi_adjust_logit = self.dropout(roi_adjust_logit)
        token_roi = roi_adjust(token_roi, roi_adjust_logit)
        return token_roi

    def sampling_mixing(self, img_feat: Tensor, token_embedding: Tensor, token_roi: Tensor):
        sampling_offset_adaptive = self.roi_offset_module(
            token_embedding
        )
        sampling_offset_base = self.roi_offset_bias.view(1, 1, -1).repeat(
            token_embedding.size(0), token_embedding.size(1), 1)
        sampling_offset = sampling_offset_base + sampling_offset_adaptive

        sampling_point = make_absolute_sampling_point(
            token_roi.transpose(0, 1),
            sampling_offset.transpose(0, 1),
            1,
            self.num_sampling_points
        )

        sampled_feat = sampling_from_img_feat(
            sampling_point,
            img_feat,
            n_points=self.num_sampling_points,
        )

        src = self.adaptive_mixing(
            sampled_feat, token_embedding.transpose(0, 1))
        src = src.transpose(0, 1)
        src = self.dropout(src)
        token_embedding = token_embedding + src

        return token_embedding

    def ffn_forward(self, token_embedding: Tensor):
        src = self.ffn(token_embedding)
        src = self.dropout(src)
        token_embedding = token_embedding + src
        return token_embedding

    def self_attention_forward(self, token_embedding: Tensor):
        src = self.ln_attn(token_embedding)
        src, _ = self.attn(src, src, src)
        src = self.dropout(src)
        token_embedding = token_embedding + src
        return token_embedding

    def forward_inner(self,
                      img_feat: Tensor,
                      token_embedding: Tensor,
                      token_roi: Tensor,
                      drop_path=None,):
        if drop_path is not None:
            self.dropout.drop_prob = drop_path

        token_embedding = self.self_attention_forward(token_embedding)
        if self.adjusting_enabled:
            token_roi = self.shift_token_roi(token_embedding, token_roi)
        if self.sampling_enabled:
            token_embedding = self.sampling_mixing(
                img_feat, token_embedding, token_roi)
        token_embedding = self.ffn_forward(token_embedding)

        return token_embedding, token_roi

    def forward(self,
                img_feat: Tensor,
                token_embedding: Tensor,
                token_roi: Tensor,
                drop_path=None,):
        for i in range(self.repeats):
            drop_path = self.drop_path
            _drop_path = drop_path if isinstance(
                drop_path, float) else drop_path[i]
            token_embedding, token_roi = self.forward_inner(
                img_feat,
                token_embedding,
                token_roi,
                _drop_path,
            )

        return token_embedding, token_roi


class AvgTokenHead(nn.Module):
    def __init__(self, in_dim, dim=0, num_classes=1000):
        super(AvgTokenHead, self).__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.norm = LAYER_NORM(in_dim)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=self.dim)
        return self.classifier(self.norm(x))


class EarlyConvolution(nn.Module):
    def __init__(self, conv_dim: int):
        super(EarlyConvolution, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = nn.Conv2d(3, self.conv_dim,
                               kernel_size=7, stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = layer_norm_by_dim(x, 1)
        return x


class SparseFormer(nn.Module):
    def __init__(self,
                 conv_dim=96,
                 num_latent_tokens=49,
                 token_sampling_points=36,
                 width_configurations=[256, 512],
                 block_sizes=[1, 8],
                 repeats=[4, 1],
                 drop_path_rate=0.2):
        super(SparseFormer, self).__init__()
        self.num_latent_tokens = num_latent_tokens
        self.width_configurations = width_configurations
        self.block_sizes = block_sizes
        self.repeats = repeats

        self.feat_extractor = EarlyConvolution(conv_dim=conv_dim)

        start_dim = width_configurations[0]
        end_dim = width_configurations[-1]
        self.initial_roi = nn.Embedding(num_latent_tokens, 4)
        self.initial_embedding = nn.Embedding(num_latent_tokens, start_dim)

        self.layers = nn.ModuleList()

        # Preparing the list of drop path rate
        nums = []
        for i, width in enumerate(width_configurations):
            repeat = repeats[i]
            block_size = block_sizes[i]
            nums += [block_size * repeat]
        lin_drop_path = list(torch.linspace(0.0, drop_path_rate, sum(nums)))
        lin_drop_path = [p.item() for p in lin_drop_path]

        block_wise_idx = 0
        for i, width in enumerate(width_configurations):
            repeat = repeats[i]
            block_size = block_sizes[i]
            nums += [block_size * repeat]
            if i > 0 and width_configurations[i-1] != width_configurations[i]:
                transition = nn.Sequential(
                    nn.Linear(
                        width_configurations[i-1], width_configurations[i]),
                    LAYER_NORM(width_configurations[i])
                )
                self.layers.append(transition)

            if repeat > 1:
                assert block_size == 1

            for block_idx in range(block_size):
                is_leading_block = (block_idx == 0)
                module = SFUnit(
                    width,
                    conv_dim=conv_dim,
                    num_sampling_points=token_sampling_points,
                    sampling_enabled=is_leading_block,
                    adjusting_enabled=is_leading_block,
                    repeats=repeat,
                    drop_path=lin_drop_path[block_wise_idx:block_wise_idx+repeat]
                )
                self.layers.append(module)
                block_wise_idx += repeat

        self.head = AvgTokenHead(end_dim, dim=0)

        self.srnet_init()

    def srnet_init(self):
        # first recursively initialize transformer related weights
        def _init_transformers_weights(m):
            if isinstance(m, nn.Linear):
                init_linear_params(m.weight, m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_transformers_weights)

        # then special case
        for n, m in self.named_modules():
            if hasattr(m, 'init_layer'):
                type_ = str(type(m))
                m.init_layer()

    def no_weight_decay(self):
        return ['token_roi']

    @torch.no_grad()
    def init_layer(self):
        nn.init.trunc_normal_(self.initial_embedding.weight, std=1.)

        self.initial_roi.weight[..., :2].data.fill_(0.0)
        self.initial_roi.weight[..., 2:].data.fill_(1.0)

        def init_rois_to_grid(root, offset, unit_width=0.5):
            grid = torch.arange(root).float()/(root-1)
            grid = grid.view(root, -1)
            grid_x = grid.view(root, 1, 1).repeat(1, root, 1)
            grid_y = grid.view(1, root, 1).repeat(root, 1, 1)
            grid = torch.cat([grid_x, grid_y], dim=-1)
            token_roi = self.initial_roi.weight[offset:offset+root**2].view(
                root, root, -1)
            token_roi.data[..., 0:2] = grid * (1-unit_width) + 0.00
            token_roi.data[..., 2:4] = grid * (1-unit_width) + unit_width

        root = int(self.num_latent_tokens**0.5)
        init_rois_to_grid(root, 0, 0.5)

    def forward(self, x: Tensor, return_embedding_and_roi: bool = False, scale=1.):
        RestrictGradNorm.GRAD_SCALE = scale
        img_feat = self.feat_extractor(x)
        img_feat = _maybe_promote(img_feat)

        batch_size = img_feat.size(0)

        token_embedding = self.initial_embedding.weight\
            .unsqueeze(1).repeat(1, batch_size, 1)
        token_roi = self.initial_roi.weight\
            .unsqueeze(1).repeat(1, batch_size, 1)

        token_embedding = _maybe_promote(token_embedding)
        token_roi = _maybe_promote(token_roi)

        for layer in self.layers:
            if isinstance(layer, SFUnit):
                token_embedding, token_roi = layer(
                    img_feat, token_embedding, token_roi)
            else:
                token_embedding = layer(token_embedding)
        if return_embedding_and_roi:
            return self.head.norm(token_embedding), token_roi
        else:
            return self.head(token_embedding)


if __name__ == '__main__':
    net = SparseFormer()

    net.eval()

    IMG_SIZE = 224
    input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(net, input)
    print(flop_count_table(flops, max_depth=2))
