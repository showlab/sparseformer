import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.cuda.amp import autocast

from .modelling import SparseFormer



@autocast(enabled=False)
def roi_adjust_3d(token_roi: Tensor, token_adjust: Tensor):
    token_xy = (token_roi[..., 3:]+token_roi[..., :3])*0.5
    token_wh = (token_roi[..., 3:]-token_roi[..., :3]).abs()

    token_xy = token_xy + token_adjust[..., :3]*token_wh
    token_wh = token_wh * token_adjust[..., 3:].exp()

    token_roi_new = torch.cat(
        [token_xy-0.5*token_wh, token_xy+0.5*token_wh], dim=-1)
    return token_roi_new


@autocast(enabled=False)
def make_absolute_sampling_point_3d(
        token_roi: Tensor,
        sampling_offset: Tensor,
        num_heads: int,
        num_points: int,
        eps=1e-5) -> Tensor:
    batch, num_tokens, _ = sampling_offset.shape

    sampling_offset = sampling_offset.reshape(batch, num_tokens, 1, num_heads*num_points, 3)

    offset_mean = sampling_offset.mean(-2, keepdim=True)
    offset_std = sampling_offset.std(-2, keepdim=True)+eps
    sampling_offset = (sampling_offset - offset_mean)/(3*offset_std)

    token_roi_xyz = (token_roi[:, :, 3:]+token_roi[:, :, :3])/2.0
    token_roi_wht = token_roi[:, :, 3:]-token_roi[:, :, :3]

    sampling_offset = sampling_offset[..., :3] * token_roi_wht.view(batch, num_tokens, 1, 1, 3)
    sampling_absolute = token_roi_xyz.reshape(batch, num_tokens, 1, 1, 3) + sampling_offset # [B, N, 1, P, 3]

    return sampling_absolute


@autocast(enabled=False)
def sampling_from_img_feat_3d(
        sampling_point: Tensor,
        img_feat: Tensor,
        n_points: int,
        restrict_grad_norm:bool=True,
        grid_sample_config:dict=None
    ):
    assert sampling_point.dtype == torch.float and img_feat.dtype == torch.float

    batch, n_tokens, one, n_points, three = sampling_point.shape
    assert one == 1 and three == 3
    batch, channel, temporal, height, width = img_feat.shape


    sampling_point = sampling_point.reshape(batch, 1, n_tokens, n_points, 3)

    sampling_point = sampling_point*2.0-1.0
    img_feat = img_feat.view(batch, channel, temporal, height, width)

    grid_sample_config = dict(mode="trilinear", padding_mode="border", align_corners=False) if grid_sample_config is None else grid_sample_config

    out = F.grid_sample(
        img_feat, sampling_point,
        **grid_sample_config
    )

    out = out.reshape(batch, channel, n_tokens, n_points)

    return out.permute(0, 2, 3, 1).unsqueeze(2) # [B, n_tokens, 1, n_points, channels]


@torch.no_grad()
def position_encoding_3d(roi: Tensor, embed: Tensor, max_temperature=128):
    # TODO add temporal PE
    return torch.zeros_like(embed)
    assert roi.size(-1) == 6
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


class MediaSparseFormer(SparseFormer):
    def __init__(self,
        replicates=1,
        **kwargs,
    ):
        kwargs.update(
            sampling_ops=[roi_adjust_3d, make_absolute_sampling_point_3d, sampling_from_img_feat_3d, position_encoding_3d]
        )
        super(MediaSparseFormer, self).__init__(**kwargs)

        self.num_latent_tokens = self.num_latent_tokens*replicates
        self.initial_roi = nn.Embedding(self.num_latent_tokens, 6)
        if self.initial_embedding.weight.size(0) != 1:
            self.initial_embedding = nn.Embedding(self.num_latent_tokens, self.width_configurations[0])

        conv1 = self.feat_extractor.conv1
        maxpool = self.feat_extractor.maxpool

        self.feat_extractor.conv1 = nn.Conv3d(
            conv1.in_channels,
            conv1.out_channels,
            kernel_size=(1,)+conv1.kernel_size,
            stride=(1,)+conv1.stride,
            padding=(0,)+conv1.padding
        )

        self.feat_extractor.maxpool = nn.MaxPool3d(
            kernel_size=(1,3,3),
            stride=(1,2,2),
            padding=(0,1,1)
        )

        for m in self.layers:
            if hasattr(m, "roi_offset_module") and m.roi_offset_module is not None:
                m.roi_offset_module[-1] = nn.Linear(m.dim, m.num_sampling_points*3, bias=False)
                m.roi_offset_bias = nn.Parameter(torch.randn(m.num_sampling_points*3))

            if hasattr(m, "roi_adjust_module") and m.roi_adjust_module is not None:
                layer = m.roi_adjust_module[-1]
                m.roi_adjust_module[-1] = nn.Linear(layer.in_features, 6, bias=False)


    def load_2d_state_dict(self, state_dict, **kwargs):
        required = self.state_dict()
        to_load = dict()
        keys = state_dict.keys()
        for key in keys:
            required_shape = required[key].shape
            existing_shape = state_dict[key].shape
            tensor = state_dict[key]
            if required_shape != existing_shape:
                if "feat_extractor.conv1" in key:
                    tensor = tensor.unsqueeze(2)
                elif "initial_roi.weight" in key:
                    ratio = required_shape[0]//existing_shape[0]
                    tensor = tensor.reshape(1, existing_shape[0], 4)
                    lt, rb = tensor.chunk(2, dim=2)
                    tm = torch.linspace(0.0+0.5/ratio, 1.0-0.5/ratio, steps=ratio).reshape(ratio, 1, 1)
                    tl = 0.5/ratio
                    lt = lt.repeat(ratio, 1, 1)
                    rb = rb.repeat(ratio, 1, 1)
                    jk = torch.cat([tm-tl, tm+tl], dim=2).repeat(1, tensor.size(1), 1)
                    j, k = jk.chunk(2, dim=2)
                    tensor = torch.cat([lt, j, rb, k], dim=2).flatten(0, 1)
                elif "initial_embedding.weight" in key:
                    ratio = required_shape[0]//existing_shape[0]
                    tensor = tensor.reshape(1, existing_shape[0], existing_shape[1]).repeat(ratio, 1, 1).flatten(0, 1)
                elif "roi_offset_bias" in key:
                    tensor = tensor.reshape(-1, 2)
                    zeros = torch.zeros_like(tensor[:, :1])
                    tensor = torch.cat([tensor, zeros], dim=1).flatten(0, 1)
                elif "roi_offset_module" in key and "weight" in key:
                    tensor = tensor.reshape(-1, 2, existing_shape[1])
                    zeros = torch.zeros_like(tensor[:, :1])
                    tensor = torch.cat([tensor, zeros], dim=1).flatten(0, 1)
                elif "roi_adjust_module" in key:
                    lt, rb = tensor.chunk(2, dim=0)
                    zeros = torch.zeros_like(lt[:1])
                    tensor = torch.cat([lt, zeros, rb, zeros], dim=0)

            to_load[key] = tensor

        return self.load_state_dict(to_load, **kwargs)
