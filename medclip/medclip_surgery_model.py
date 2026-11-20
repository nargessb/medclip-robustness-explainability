# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:00:33 2025

@author: narges.babadi1
"""

from collections import OrderedDict
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from medclip import MedCLIPVisionModelViT
import numpy as np
import torch
from torch import nn
import torch
from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        k = v
        q = v

        if self.settings == 'resnet':
            k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            q = k
            scale = self.scale * 8
        else:
            scale = self.scale

        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]





class AttentionPool2d_MedCLIP(nn.Module):
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spatial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self.attn = None
        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

    def forward(self, x):
        # Lazy surgery to replace Q/K with V after loading weights
        if self.attn is None:
            from medclip_surgery.attention import Attention  # use your local Attention class
            self.attn = Attention(self.output_dim, self.embed_dim, self.num_heads, qkv_bias=True)
            self.attn.qkv.weight.data = torch.cat([
                self.v_proj.weight.data,
                self.v_proj.weight.data,
                self.v_proj.weight.data
            ], dim=0)
            self.attn.qkv.bias.data = torch.cat([
                self.v_proj.bias.data,
                self.v_proj.bias.data,
                self.v_proj.bias.data
            ], dim=0)
            self.attn.proj.weight.data = self.c_proj.weight.data
            self.attn.proj.bias.data = self.c_proj.bias.data

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(2, 0, 1)  # NCHW -> (HW)NC
        cls_token = x.mean(dim=0, keepdim=True)
        x = torch.cat([cls_token, x], dim=0)  # (HW+1)NC

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[0] - 1) ** 0.5)

        if side != new_side:
            pos = self.positional_embedding[1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
            pos = nn.functional.interpolate(pos, (new_side, new_side), mode="bilinear")
            pos = pos.reshape(-1, new_side * new_side).permute(1, 0)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1], pos], dim=0)

        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, x_ori = self.attn(x.transpose(0, 1))  # Apply attention with v-v path
        x[:, 0, :] = x_ori[:, 0, :]  # Replace class token with original branch
        return x


class ModifiedResNet_MedCLIP(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # Stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # Residual layers
        from medclip_surgery.bottleneck import Bottleneck  # use your local Bottleneck
        self._inplanes = width
        self.layer1 = self._make_layer(Bottleneck, width, layers[0])
        self.layer2 = self._make_layer(Bottleneck, width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d_MedCLIP(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self._inplanes, planes, stride)]
        self._inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock_MedCLIP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        # dual path logic for attention
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res
                return [x, x_ori]

            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # single path fallback
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x

class Transformer_MedCLIP(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock_MedCLIP(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer_MedCLIP(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_MedCLIP(width, layers, heads)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.no_grad()
    def forward(self, x):
        if self.attn is None:
            for i in range(1, 7):
                self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype).unsqueeze(0).repeat(x.size(0), 1, 1),
            x
        ], dim=1)

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(side, side, x.shape[-1]).permute(2, 0, 1).unsqueeze(0)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear', align_corners=False)
            new_pos = new_pos.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, x.shape[-1])
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos], 0)

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x, x_ori = self.transformer(x)
        x[0, :, :] = x_ori[0, :, :]
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)
        x = x @ self.proj
        return x




class MedCLIPSurgery(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 vision_model_path: str = None,
                 text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 image_resolution: int = 224):
        super().__init__()

        # Vision Encoder
        self.visual = MedCLIPVisionModelViT()
        if vision_model_path:
            state_dict = torch.load(vision_model_path, map_location="cpu")
            self.visual.load_state_dict(state_dict, strict=False)

        # Text Encoder (Clinical BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_embed_dim = self.text_encoder.config.hidden_size

        # Projection layers
        self.text_projection = nn.Linear(self.text_embed_dim, embed_dim)
        self.image_projection = nn.Linear(768, embed_dim)  # or 1024 if using ViT-L, etc.

        # Logit scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, image):
        image_features = self.visual(image)  # shape: (B, D)
        image_features = self.image_projection(image_features)  # shape: (B, embed_dim)
        return image_features

    def encode_text(self, text_inputs):
        outputs = self.text_encoder(**text_inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        text_features = self.text_projection(cls_embeddings)  # shape: (B, embed_dim)
        return text_features

    def forward(self, image, text_inputs):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
