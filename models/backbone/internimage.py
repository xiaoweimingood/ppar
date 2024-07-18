# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import yaml
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
from models.ops.ops_dcnv3 import modules as opsm
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from torch.nn.modules.transformer import _get_activation_fn
from models.registry import BACKBONE

class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MLDecoder(nn.Module):
    def __init__(self, 
                 nattr, 
                 c_in, 
                 bn=False, 
                 pool='avg', 
                 scale=30, 
                 num_of_groups=-1, 
                 decoder_embedding=768,
                 zsl=0):
    # def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768,
    #              initial_num_features=2048, zsl=0):
        super(MLDecoder, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > nattr:
            embed_len_decoder = nattr

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(c_in, decoder_embedding)

        # non-learnable queries
        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            query_embed.requires_grad_(False)
        else:
            query_embed = None

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = nattr
            self.decoder.duplicate_factor = int(nattr / embed_len_decoder + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(nattr))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x, label=None):
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.decoder.query_embed))
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        # return logits, embedding_spatial_786
        return logits

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class CrossAttention(nn.Module):
    r""" Cross Attention Module
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        attn_head_dim (int, optional): Dimension of attention head.
        out_dim (int, optional): Dimension of output.
    """
    
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):
    r"""Attentive Block
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        attn_head_dim (int, optional): Dimension of attention head. Default: None.
        out_dim (int, optional): Dimension of output. Default: None.
    """
    
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer="LN",
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()

        self.norm1_q = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_k = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_v = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.cross_dcn = CrossAttention(dim,
                                        num_heads=num_heads,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        attn_drop=attn_drop,
                                        proj_drop=drop,
                                        attn_head_dim=attn_head_dim,
                                        out_dim=out_dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x_q,
                x_kv,
                pos_q,
                pos_k,
                bool_masked_pos,
                rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)

        x = self.cross_dcn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv = x
        pos_q, pos_k = 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k,
                            bool_masked_pos=None,
                            rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False, # for InternImage-H/G
                 remove_center=False,  # for InternImage-H/G
                 ):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale, # for InternImage-H/G
            remove_center=remove_center,  # for InternImage-H/G
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm: # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 post_norm_block_ids=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False, # for InternImage-H/G
                 remove_center=False,  # for InternImage-H/G
                 ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale, # for InternImage-H/G
                remove_center = remove_center,  # for InternImage-H/G
        ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None: # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [build_norm_layer(channels, 'LN', eps=1e-6) for _ in post_norm_block_ids]
            )
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x) # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


class InternImage(nn.Module):
    r""" InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        num_classes (int): Number of classes. Default: 1000
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        dw_kernel_size (int): Size of the dwconv. Default: None
        use_clip_projector (bool): Whether to use clip projector. Default: False
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """

    def __init__(self,
                 core_op='DCNv3',
                 channels=64,
                 depths=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 num_classes=1000,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 cls_scale=1.5,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 use_clip_projector=False, # for InternImage-H/G
                 level2_post_norm=False, # for InternImage-H/G
                 level2_post_norm_block_ids=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False, # for InternImage-H/G
                 remove_center=False, # for InternImage-H/G
                 **kwargs):
        super().__init__()
        self.core_op = core_op
        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2**(self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.use_clip_projector = use_clip_projector
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        self.remove_center = remove_center

        # print(f'using core type: {core_op}')
        # print(f'using activation layer: {act_layer}')
        # print(f'using main norm layer: {norm_layer}')
        # print(f'using dpr: {drop_path_type}, {drop_path_rate}')
        # print(f"level2_post_norm: {level2_post_norm}")
        # print(f"level2_post_norm_block_ids: {level2_post_norm_block_ids}")
        # print(f"res_post_norm: {res_post_norm}")
        # print(f"remove_center: {remove_center}")

        in_chans = 3
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                i == 2) else None # for InternImage-H/G
            level = InternImageBlock(
                core_op=getattr(opsm, core_op),
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale, # for InternImage-H/G
                remove_center=remove_center,  # for InternImage-H/G
            )
            self.levels.append(level)
        
        if not use_clip_projector: # for InternImage-T/S/B/L/XL
            self.conv_head = nn.Sequential(
                nn.Conv2d(self.num_features,
                          int(self.num_features * cls_scale),
                          kernel_size=1,
                          bias=False),
                build_norm_layer(int(self.num_features * cls_scale), 'BN',
                                 'channels_first', 'channels_first'),
                build_act_layer(act_layer))
            # self.head = nn.Linear(int(self.num_features * cls_scale), num_classes) 
            #     if num_classes > 0 else nn.Identity()
            # self.head = MLDecoder(88, int(self.num_features * cls_scale))
            # self.head = MLDecoder(88, int(self.num_features))
        else: # for InternImage-H/G
            pretrain_embed_dim, _stride, attnpool_num_heads, clip_embed_dim = 1024, 2, 16, 768
            self.dcnv3_head_x4 = nn.Sequential(
                nn.Conv2d(in_channels=self.num_features,
                          out_channels=pretrain_embed_dim * (_stride ** 2),
                          kernel_size=1), nn.PixelShuffle(_stride))
            self.dcnv3_head_x3 = nn.Conv2d(in_channels=self.num_features // 2,
                                           out_channels=pretrain_embed_dim,
                                           kernel_size=1)
            self.clip_projector = AttentionPoolingBlock(
                dim=pretrain_embed_dim,
                num_heads=attnpool_num_heads,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                norm_layer=norm_layer,
                out_dim=clip_embed_dim)
            self.fc_norm = build_norm_layer(clip_embed_dim, norm_layer, eps=1e-6)
            self.head = nn.Linear(
                clip_embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(opsm, self.core_op)):
            m._reset_parameters()

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = 'levels.{}.blocks.{}.'.format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios['levels.0.blocks.0.']
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios['levels.1.blocks.0.']
        lr_ratios["levels.0.norm"] = lr_ratios['levels.1.blocks.0.']
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios['levels.2.blocks.0.']
        lr_ratios["levels.1.norm"] = lr_ratios['levels.2.blocks.0.']
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios['levels.3.blocks.0.']
        lr_ratios["levels.2.norm"] = lr_ratios['levels.3.blocks.0.']
        return lr_ratios

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.conv_head(x.permute(0, 3, 1, 2))
        # x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x

    def forward_features_seq_out(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level in self.levels:
            x, x_ = level(x, return_wo_downsample=True)
            seq_out.append(x_)
        return seq_out
    
    def forward_clip_projector(self, x): # for InternImage-H/G
        xs = self.forward_features_seq_out(x)
        x1, x2, x3, x4 = xs
        
        x1 = x1.permute(0, 3, 1, 2) # NHWC -> NCHW
        x2 = x2.permute(0, 3, 1, 2) # NHWC -> NCHW
        x3 = x3.permute(0, 3, 1, 2) # NHWC -> NCHW
        x4 = x4.permute(0, 3, 1, 2) # NHWC -> NCHW

        x4 = self.dcnv3_head_x4(x4)
        x = x4
        x3 = self.dcnv3_head_x3(x3)
        x = x + x3

        x = x.flatten(-2).transpose(1, 2).contiguous()
        x = self.clip_projector(x)
        x = self.fc_norm(x)
        
        return x
    
    def forward(self, x):
        if self.use_clip_projector: # for InternImage-H/G
            x = self.forward_clip_projector(x)
        else: # for InternImage-T/S/B/L/XL
            x = self.forward_features(x)
        # x = self.head(x)
        return x

@BACKBONE.register("intern_image_b")
def internimage_b(pretrained=False, 
                  config_path='models/configs/internimage_b_1k_224.yaml', 
                  **kwargs):
    num_classes = kwargs.pop("num_classes", 88)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = build_config_recursive(config)
    print(config)
    model_type = config.MODEL.TYPE
    if model_type == 'intern_image':
        model = InternImage(
            core_op=config.MODEL.INTERN_IMAGE.CORE_OP,
            num_classes=num_classes,
            channels=config.MODEL.INTERN_IMAGE.CHANNELS,
            depths=config.MODEL.INTERN_IMAGE.DEPTHS,
            groups=config.MODEL.INTERN_IMAGE.GROUPS,
            layer_scale=config.MODEL.INTERN_IMAGE.LAYER_SCALE,
            offset_scale=config.MODEL.INTERN_IMAGE.OFFSET_SCALE,
            post_norm=config.MODEL.INTERN_IMAGE.POST_NORM,
            mlp_ratio=config.MODEL.INTERN_IMAGE.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            res_post_norm=config.MODEL.INTERN_IMAGE.RES_POST_NORM, # for InternImage-H/G
            dw_kernel_size=config.MODEL.INTERN_IMAGE.DW_KERNEL_SIZE, # for InternImage-H/G
            use_clip_projector=config.MODEL.INTERN_IMAGE.USE_CLIP_PROJECTOR, # for InternImage-H/G
            level2_post_norm=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM, # for InternImage-H/G
            level2_post_norm_block_ids=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM_BLOCK_IDS, # for InternImage-H/G
            center_feature_scale=config.MODEL.INTERN_IMAGE.CENTER_FEATURE_SCALE, # for InternImage-H/G
            remove_center=config.MODEL.INTERN_IMAGE.REMOVE_CENTER,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

@BACKBONE.register("intern_image_t")
def internimage_t(pretrained=False, 
                  config_path='models/configs/internimage_t_1k_224.yaml', 
                  **kwargs):
    num_classes = kwargs.pop("num_classes", 88)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = build_config_recursive(config)
    print(config)
    model_type = config.MODEL.TYPE
    if model_type == 'intern_image':
        model = InternImage(
            core_op=config.MODEL.INTERN_IMAGE.CORE_OP,
            num_classes=num_classes,
            channels=config.MODEL.INTERN_IMAGE.CHANNELS,
            depths=config.MODEL.INTERN_IMAGE.DEPTHS,
            groups=config.MODEL.INTERN_IMAGE.GROUPS,
            layer_scale=config.MODEL.INTERN_IMAGE.LAYER_SCALE,
            offset_scale=config.MODEL.INTERN_IMAGE.OFFSET_SCALE,
            post_norm=config.MODEL.INTERN_IMAGE.POST_NORM,
            mlp_ratio=config.MODEL.INTERN_IMAGE.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            res_post_norm=config.MODEL.INTERN_IMAGE.RES_POST_NORM, # for InternImage-H/G
            dw_kernel_size=config.MODEL.INTERN_IMAGE.DW_KERNEL_SIZE, # for InternImage-H/G
            use_clip_projector=config.MODEL.INTERN_IMAGE.USE_CLIP_PROJECTOR, # for InternImage-H/G
            level2_post_norm=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM, # for InternImage-H/G
            level2_post_norm_block_ids=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM_BLOCK_IDS, # for InternImage-H/G
            center_feature_scale=config.MODEL.INTERN_IMAGE.CENTER_FEATURE_SCALE, # for InternImage-H/G
            remove_center=config.MODEL.INTERN_IMAGE.REMOVE_CENTER,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    state_dict = torch.load('/public/share/others/challenge_dataset/weights/internimage_t_1k_224.pth', 
                            map_location='cpu')
    print(state_dict.keys())
    
    model.load_state_dict(state_dict['model'], strict=False)

    return model

def build_config_recursive(config):
    node = Config()
    if not isinstance(config, dict):
        return config
    else:
        for k, v in config.items():
            # node.__setattr__(k, build_config_recursive(v))
            setattr(node, k, build_config_recursive(v))
        return node 
    
class Config():
    def __init__(self) -> None:
        pass