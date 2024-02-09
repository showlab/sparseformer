# --------------------------------------------------------
# SparseFormer
# Copyright 2023 Ziteng Gao
# Licensed under The MIT License
# Written by Ziteng Gao
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from enum import Enum

from torch import Tensor
from torch.cuda.amp import autocast
from .utils import _maybe_promote, init_layer_norm_unit_norm, init_linear_params, RestrictGradNorm, DropPath, CompatibleAttrDict


from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Attention

LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)

class OP(Enum):
    ATTN = 1
    MLP = 2
    SAMPLING_M = 3
    ROI_ADJ = 4
    PE_INJECT = 5

DEFAULT_OPS_LIST = \
[[OP.ATTN, OP.ROI_ADJ, OP.SAMPLING_M, OP.MLP]] + \
[[OP.ATTN, OP.ROI_ADJ, OP.SAMPLING_M, OP.MLP]] + \
[[OP.ATTN, OP.MLP]] * 7


@autocast(enabled=False)
def roi_adjust(token_roi: Tensor, token_adjust: Tensor):
    token_xy = (token_roi[..., 2:]+token_roi[..., :2])*0.5
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
        num_points: int,
        eps=1e-5) -> Tensor:
    batch, num_tokens, _ = sampling_offset.shape

    sampling_offset = sampling_offset.view(batch, num_tokens, 1, num_heads*num_points, 2)

    offset_mean = sampling_offset.mean(-2, keepdim=True)
    offset_std = sampling_offset.std(-2, keepdim=True)+eps
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
        n_points: int,
        restrict_grad_norm:bool=True,
        grid_sample_config:dict=None
    ):
    batch, Hq, Wq, num_points_per_head, _ = sampling_point.shape
    batch, channel, height, width = img_feat.shape

    n_heads = num_points_per_head//n_points

    sampling_point = sampling_point.view(batch, Hq, Wq, n_heads, n_points, 2) \
        .permute(0, 3, 1, 2, 4, 5).contiguous().flatten(0, 1)

    if restrict_grad_norm:
        # We truncate the grad for sampling coordinates to the unit length (e.g., 1.0/height)
        # to avoid inaccurate gradients due to bilinear sampling. That is, we restrict
        # gradients to be local.
        sampling_point = RestrictGradNorm.apply(sampling_point, 1.0/height)

    sampling_point = sampling_point.flatten(2, 3)
    sampling_point = sampling_point*2.0-1.0
    img_feat = img_feat.view(batch*n_heads, channel//n_heads, height, width)

    grid_sample_config = dict(mode="bilinear", padding_mode="border", align_corners=False) if grid_sample_config is None else grid_sample_config
    out = F.grid_sample(
        img_feat, sampling_point,
        **grid_sample_config
    )

    out = out.view(batch, n_heads, channel//n_heads, Hq, Wq, n_points)

    return out.permute(0, 3, 4, 1, 5, 2).flatten(1, 2)

@torch.no_grad()
def position_encoding(roi: Tensor, embed: Tensor, max_temperature=128):
    assert roi.size(-1) == 4
    num_feats = embed.size(-1) // 4
    roi = roi * math.pi
    dim_t = torch.linspace(
        0, math.log(max_temperature), num_feats, dtype=roi.dtype, device=roi.device
    )
    dim_t = dim_t.exp().view(1, 1, 1, -1)
    pos_x = roi[..., None] * dim_t
    pos_x = torch.cat(
        (
            pos_x[..., 0::2].sin() ,
            pos_x[..., 1::2].cos() ,
        ),
        dim=3,
    ).flatten(2)
    return pos_x

@autocast(enabled=False)
def layer_norm_by_dim(x: Tensor, dim: int=-1, eps=1e-5):
    var, mean = torch.var_mean(x, dim=dim, unbiased=False, keepdim=True)
    return (x - mean) / (var + eps).sqrt()


class QuickGELU(nn.Module):
    # NOTE borrowed from openclip
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AdaptiveMixing(nn.Module):
    def __init__(self,
            in_dim: int,
            in_points: int,
            query_dim: int=None,
            out_dim: int=None,
            out_points: int=None,
            out_query_dim: int=None,
            mixing_bias: bool = True,
            pg_inner_dim = .25,
            norm_layer = nn.LayerNorm
        ):
        super(AdaptiveMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        out_points = max(out_points, 32)
        query_dim = query_dim if query_dim is not None else in_dim
        out_query_dim = out_query_dim if out_query_dim is not None else query_dim

        self.query_dim = query_dim
        self.out_query_dim = out_query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = self.in_dim
        self.eff_out_dim = self.out_dim

        self.mixing_bias = mixing_bias

        self.channel_param_count = (self.eff_in_dim * self.eff_out_dim)
        self.spatial_param_count = (self.in_points * self.out_points)

        self.total_param_count = self.channel_param_count + self.spatial_param_count

        if isinstance(pg_inner_dim, float):
            pg_inner_dim = int(pg_inner_dim*query_dim)
        else:
            assert isinstance(pg_inner_dim, int)

        self.parameter_generator = nn.Sequential(
            norm_layer(self.query_dim),
            nn.Linear(self.query_dim, pg_inner_dim),
            nn.Linear(pg_inner_dim, self.total_param_count),
        )

        if self.mixing_bias:
            self.m_beta = nn.Parameter(torch.zeros(self.eff_out_dim))
            self.s_beta = nn.Parameter(torch.zeros(self.out_points))

        self.out_proj = nn.Linear(self.eff_out_dim*self.out_points,
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

        if self.mixing_bias:
            channel_bias = self.m_beta.view(1, 1, self.eff_out_dim)
            spatial_bias = self.s_beta.view(1, self.out_points, 1)
            out = torch.baddbmm(channel_bias, out, channel_mixing)
            out = self.act(out)
            out = torch.baddbmm(spatial_bias, spatial_mixing, out)
        else:
            out = torch.bmm(out, channel_mixing)
            out = torch.bmm(spatial_mixing, out)

        out = self.act(out)

        out = out.reshape(batch, num_tokens, -1)
        out = self.out_proj(out)

        return out


class SFUnit(nn.Module):
    def __init__(
        self,
        dim,
        img_feat_dim,
        ops=[OP.ATTN, OP.MLP],
        num_sampling_points=36,
        mlp_ratio=4,
        repeats=1,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        compatible_config: CompatibleAttrDict = None,
    ):
        super(SFUnit, self).__init__()

        self.dim = dim
        self.ops = ops

        self.num_sampling_points = num_sampling_points
        self.repeats = repeats
        self.compatible_config = compatible_config

        if compatible_config.use_stage_embedding and repeats > 1:
            self.stage_embedding = nn.Embedding(repeats, dim)
        else:
            self.stage_embedding = None

        mlp_act_layer = QuickGELU if (repeats == 1) and compatible_config.clip_quick_gelu else nn.GELU


        if OP.SAMPLING_M in ops:
            self.adaptive_mixing = AdaptiveMixing(
                img_feat_dim, num_sampling_points, query_dim=dim, out_query_dim=dim,
                pg_inner_dim=compatible_config.pg_inner_dim,
                mixing_bias=compatible_config.mixing_bias,
            )
            self.roi_offset_module = nn.Sequential(
                norm_layer(dim),
                nn.Linear(dim, num_sampling_points*2, bias=False),
            )
            self.roi_offset_bias = nn.Parameter(
                torch.randn(num_sampling_points*2)
            )
        else:
            self.adaptive_mixing = None
            self.roi_offset_module = None
            self.roi_offset_bias = None


        if OP.ATTN in ops:
            num_attn_heads = 64 if compatible_config.type_num_attn_heads == 1 else dim//64
            self.norm1 = norm_layer(dim)
            self.attn = Attention(
                dim,
                num_attn_heads,
                qkv_bias=True,
            )

        if OP.MLP in ops:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(dim, hidden_features=dim*mlp_ratio, act_layer=mlp_act_layer)


        if OP.ROI_ADJ in ops:
            type_op_adjust = compatible_config.type_op_adjust
            if type_op_adjust == 1:
                self.roi_adjust_module = nn.Sequential(
                    norm_layer(dim),
                    nn.Linear(dim, 4, bias=False),
                )
            elif type_op_adjust == 2:
                self.roi_adjust_module = nn.Sequential(
                    norm_layer(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, 4, bias=False),
                )

        if OP.PE_INJECT in ops:
            self.pe_transition = nn.Sequential(
                norm_layer(dim),
                nn.Linear(dim, dim),
            )

        self.drop_path = drop_path
        self.dropout = DropPath(
            drop_path if isinstance(drop_path, (float, int)) else .0,
            batch_first=True,
        ) 

    @torch.no_grad()
    def init_layer(self):
        ops = self.ops
        if OP.ATTN in ops:
            init_linear_params(self.attn.qkv.weight, self.attn.qkv.bias)

        if OP.ROI_ADJ in ops:
            init_layer_norm_unit_norm(self.roi_adjust_module[0],  1.)
            nn.init.zeros_(self.roi_adjust_module[-1].weight)
            pass

        if OP.SAMPLING_M in ops:
            init_layer_norm_unit_norm(self.roi_offset_module[0],  1.)
            nn.init.zeros_(self.roi_offset_module[-1].weight)

            root = int(self.num_sampling_points**0.5)
            sampling_points = self.roi_offset_bias.view(-1, 2)
            if root ** 2 == self.num_sampling_points:
                x = torch.linspace(0.5/root, 1-0.5/root, root)\
                    .view(1, -1, 1).repeat(root, 1, 1)
                y = torch.linspace(0.5/root, 1-0.5/root, root)\
                    .view(-1, 1, 1).repeat(1, root, 1)
                grid = torch.cat([x, y], dim=-1).view(root**2, -1)
                sampling_points.data[:] = grid - 0.5
            else:
                nn.init.uniform_(sampling_points, -0.5, 0.5)

    def shift_token_roi(self, inp: Tensor, roi: Tensor):
        logit = self.roi_adjust_module(inp)
        if self.compatible_config.type_op_adjust == 1:
            logit = self.dropout(logit)
        roi = roi_adjust(roi, logit)
        return roi

    def sampling_mixing(self, img_feat: Tensor, inp: Tensor, roi: Tensor):
        sampling_offset_adaptive = self.roi_offset_module(inp)
        sampling_offset_base = self.roi_offset_bias.view(1, 1, -1).repeat(
            inp.size(0), inp.size(1), 1)
        sampling_offset = sampling_offset_base + sampling_offset_adaptive

        sampling_point = make_absolute_sampling_point(
            roi,
            sampling_offset,
            1,
            self.num_sampling_points
        )

        sampled_feat = sampling_from_img_feat(
            sampling_point,
            img_feat,
            n_points=self.num_sampling_points,
            restrict_grad_norm=self.compatible_config.restrict_grad_norm if self.compatible_config else True,
            grid_sample_config=self.compatible_config.grid_sample_config
        )

        src = self.adaptive_mixing(sampled_feat, inp)
        src = self.dropout(src)
        inp = inp + src
        return inp


    def attn_forward(self, inp: Tensor):
        src = self.norm1(inp)
        src = self.attn(src)
        src = self.dropout(src)
        inp = inp + src
        return inp

    def mlp_forward(self, inp: Tensor):
        src = self.norm2(inp)
        src = self.mlp(src)
        src = self.dropout(src)
        inp = inp + src
        return inp

    def forward_inner(
            self,
            img_feat: Tensor,
            embed: Tensor,
            roi: Tensor,
            drop_path=None
    ):
        if drop_path is not None:
            self.dropout.drop_prob = drop_path

        for op in self.ops:
            if op == OP.ATTN:
                embed = self.attn_forward(embed)
            elif op == OP.MLP:
                embed = self.mlp_forward(embed)
            elif op == OP.ROI_ADJ:
                roi = self.shift_token_roi(embed, roi)
            elif op == OP.SAMPLING_M:
                embed = self.sampling_mixing(img_feat, embed, roi)
            elif op == OP.PE_INJECT:
                pe = self.pe_transition(position_encoding(roi, embed))
                embed = embed + pe


        return embed, roi

    def forward(
            self,
            img_feat: Tensor,
            token_embedding: Tensor,
            token_roi: Tensor,
            drop_path=None,
    ):
        for i in range(self.repeats):
            if self.stage_embedding is not None:
                token_embedding = token_embedding + self.stage_embedding.weight[i].reshape(1, 1, -1)
            drop_path = self.drop_path
            drop_path_item = drop_path if isinstance(
                drop_path, float) else drop_path[i]
            token_embedding, token_roi = self.forward_inner(
                img_feat,
                token_embedding,
                token_roi,
                drop_path_item,
            )

        return token_embedding, token_roi


class Head(nn.Module):
    def __init__(self,
            op="avg",
    ):
        super(Head, self).__init__()
        self.op = op

    def forward(self, x):
        if self.op == "avg":
            x = x.mean(1)
        elif self.op == "cls":
            x = x[:, 0]
        return x


class EarlyConvolution(nn.Module):
    def __init__(self, conv_dim: int):
        super(EarlyConvolution, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = nn.Conv2d(3, self.conv_dim, kernel_size=7, stride=2, padding=3, bias=True)
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
    def __init__(
            self,
            conv_dim=96,
            num_latent_tokens=49,
            same_token_embedding=True,
            cls_token_insert_layer=-1,
            num_sampling_points=36,
            width_configs=[256,]+[512,]*8,
            repeats=[4,]+[1,]*8,
            drop_path_rate=0.,
            ops_list=None,
            head_op="avg",
            proj_dim=1000,
            proj_bias=True,
            compatible_config: CompatibleAttrDict=None,
    ):
        super(SparseFormer, self).__init__()
        self.num_latent_tokens = num_latent_tokens
        self.width_configurations = width_configs
        self.repeats = repeats

        compatible_config = CompatibleAttrDict() if compatible_config is None else compatible_config
        norm_layer = partial(nn.LayerNorm, eps=compatible_config.ln_eps)
        ops_list = DEFAULT_OPS_LIST if ops_list is None else ops_list

        self.feat_extractor = EarlyConvolution(conv_dim=conv_dim)

        start_dim = width_configs[0]
        end_dim = width_configs[-1]
        self.initial_roi = nn.Embedding(num_latent_tokens, 4)
        if same_token_embedding:
            self.initial_embedding = nn.Embedding(1, start_dim) 
        else:
            self.initial_embedding = nn.Embedding(num_latent_tokens, start_dim) 
        
        self.cls_token_insert_idx = cls_token_insert_layer
        if cls_token_insert_layer > 0:
            self.cls_embedding = nn.Embedding(1, end_dim)
        else:
            self.cls_embedding = None

        self.layers = nn.ModuleList()

        # Preparing the list of drop path rate
        nums = []
        for i, width in enumerate(width_configs):
            repeat = repeats[i]
            nums += [repeat]
        lin_drop_path = list(torch.linspace(0.0, drop_path_rate, sum(nums)))
        lin_drop_path = [p.item() for p in lin_drop_path]

        block_wise_idx = 0
        for i, width in enumerate(width_configs):
            repeat = repeats[i]
            if i > 0 and width_configs[i-1] != width_configs[i]:
                transition = nn.Sequential(
                    nn.Linear(width_configs[i-1], width_configs[i]),
                    norm_layer(width_configs[i]) if compatible_config.transition_ln else nn.Identity()
                )
                self.layers.append(transition)

            ops = ops_list[i]

            module = SFUnit(
                width,
                img_feat_dim=conv_dim,
                num_sampling_points=num_sampling_points,
                repeats=repeat,
                drop_path=lin_drop_path[block_wise_idx:block_wise_idx+repeat],
                norm_layer=norm_layer,
                ops=ops,
                compatible_config=compatible_config
            )

            self.layers.append(module)
            block_wise_idx += repeat
        
        self.head = Head(op=head_op)
        self.norm = norm_layer(end_dim)
        self.proj = nn.Linear(end_dim, proj_dim, bias=proj_bias)

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
        return ['initial_roi']

    def _batchify(self, x: Tensor, batch_size: int):
        return x.unsqueeze(0).repeat(batch_size, 1, 1)

    def _expand_tokens(self, x: Tensor):
        return x if x.size(1) == self.num_latent_tokens else x.repeat(1, self.num_latent_tokens, 1)

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

    def forward(self, x: Tensor, scale=1., prompte_type=True):
        RestrictGradNorm.GRAD_SCALE = scale
        batch_size = x.size(0)

        img_feat = self.feat_extractor(x)

        token_embedding = self._batchify(self.initial_embedding.weight, batch_size)
        token_embedding = self._expand_tokens(token_embedding)
        token_roi = self._batchify(self.initial_roi.weight, batch_size)
        token_roi = self._expand_tokens(token_roi)

        if prompte_type:
            img_feat = _maybe_promote(img_feat)
            token_embedding = _maybe_promote(token_embedding)
            token_roi = _maybe_promote(token_roi)

        layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, SFUnit):
                if layer_idx == self.cls_token_insert_idx:
                    cls_embedding = self._batchify(self.cls_embedding.weight, batch_size)
                    token_embedding = torch.cat([cls_embedding, token_embedding], dim=1)
                token_embedding, token_roi = layer(img_feat, token_embedding, token_roi)
                layer_idx += 1
            else:
                token_embedding = layer(token_embedding)

        final = self.head(token_embedding)
        final = self.norm(final)
        final = self.proj(final)

        return final
