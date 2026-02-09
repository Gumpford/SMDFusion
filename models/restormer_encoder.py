"""Restormer-style encoder using lightweight transformer blocks."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adain import AdaIN


class FeedForward(nn.Module):
    """Gated feed-forward network used in Restormer blocks."""

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class MultiDConvHeadTrans(nn.Module):
    """Multi-head self-attention with depth-wise convolution projections."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        attn = torch.matmul(q.transpose(-2, -1), k)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        return self.project_out(out)


class RestormerBlock(nn.Module):
    """Core Restormer block with attention and feed-forward layers."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(dim, affine=True)
        self.attn = MultiDConvHeadTrans(dim, num_heads=num_heads)
        self.norm2 = nn.InstanceNorm2d(dim, affine=True)
        self.ffn = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerEncoder(nn.Module):
    """Encoder producing three scales with AdaIN prompt injection."""

    def __init__(self, channels: List[int], prompt_dim: int) -> None:
        super().__init__()
        self.embed = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
        self.stage1 = RestormerBlock(channels[0], num_heads=2)
        self.down1 = nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=4)
        self.stage2 = RestormerBlock(channels[1], num_heads=4)
        self.down2 = nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2)
        self.stage3 = RestormerBlock(channels[2], num_heads=4)
        self.adain1 = AdaIN(channels[0], prompt_dim)
        self.adain2 = AdaIN(channels[1], prompt_dim)
        self.adain3 = AdaIN(channels[2], prompt_dim)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat1 = self.stage1(self.embed(x))
        feat1 = self.adain1(feat1, prompt)
        feat2 = self.stage2(self.down1(feat1))
        feat2 = self.adain2(feat2, prompt)
        feat3 = self.stage3(self.down2(feat2))
        feat3 = self.adain3(feat3, prompt)
        return feat1, feat2, feat3
