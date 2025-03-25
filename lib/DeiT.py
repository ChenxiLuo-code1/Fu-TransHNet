# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import numpy as np


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches    #此时num_patches=196
        #print(num_patches)     #196
        #print(self.embed_dim)     #384
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, self.embed_dim))     #因为多了一个[class]token

    def forward(self, x):
        # print("*******")          #此时传进来的x的尺寸为[16,3,352,352]
        # print(x.shape)
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]      #此时B=16   x.shape[1]=3;   x.shape[2]=352


        x = self.patch_embed(x)

        # print("*******")
        # print(x.shape)       #此时x.shape=[16,484,384]


        pe = self.pos_embed    #pe.shape=[1,484,384]    因为没用到分类，所以就没加[class]token,
        # print("**###")
        # print(pe.shape)

        x = x + pe        #x.shape=[16,484,384]
        # print("**###")
        # print(x.shape)
        x = self.pos_drop(x)


        for blk in self.blocks:
            x = blk(x)
        # for blk in self.blocks:
        #     x = blk(x)

        x = self.norm(x)

        return x           #此时传进来的x的尺寸变为[16,484,384]   直接输出所有的token



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('D:/Wyy_FuTransHNet/deit_small_patch16_224_cd65a155 .pth')
        model.load_state_dict(ckpt['model'], strict=False)

    pe = model.pos_embed[:, 1:, :].detach()       #pe.shape=[1,196,384]    224/16*224*16=196
    pe = pe.transpose(-1, -2)                     #pe.shape=[1,384,196]
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))     #pe.shape=[1,384,14,14]


    pe = F.interpolate(pe, size=(22, 22), mode='bilinear', align_corners=True)   #pe.shape=[1,384,22,22]       
    pe = pe.flatten(2)    #pe.shape=[1,384,484]
    pe = pe.transpose(-1, -2)        #pe.shape=[1,484,384]
    # print('*****')
    # print(pe.shape)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg =_cfg()
    if pretrained:
        ckpt = torch.load('/home/415/wyy/test/deit_small_distilled_patch16_224-649709d9.pth')
        model.load_state_dict(ckpt['model'], strict=False)

    pe = model.pos_embed[:, 1:, :].detach()       #pe.shape=[1,196,384]    224/16*224*16=196
    pe = pe.transpose(-1, -2)                     #pe.shape=[1,384,196]
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))     #pe.shape=[1,384,14,14]


    pe = F.interpolate(pe, size=(22, 22), mode='bilinear', align_corners=True)   #pe.shape=[1,384,22,22]
    pe = pe.flatten(2)    #pe.shape=[1,384,484]
    pe = pe.transpose(-1, -2)        #pe.shape=[1,484,384]
    print('*****')
    print(pe.shape)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model