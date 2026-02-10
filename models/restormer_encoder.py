"""Restormer-style encoder using lightweight transformer blocks."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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

    def __init__(self, dim: int, num_heads: int = 4, attn_chunk_size: int | None = 1024) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_chunk_size = attn_chunk_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        outputs = []
        seq_len = q.size(-2)
        for start in range(0, seq_len, chunk_size):
            q_chunk = q[..., start : start + chunk_size, :]
            attn = torch.matmul(q_chunk, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            outputs.append(torch.matmul(attn, v))
        return torch.cat(outputs, dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        if self.attn_chunk_size and q.size(-2) > self.attn_chunk_size:
            out = self._chunked_attention(q, k, v, self.attn_chunk_size)
        else:
            out = self._full_attention(q, k, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        return self.project_out(out)


class RestormerBlock(nn.Module):
    """Core Restormer block with attention and feed-forward layers."""

    def __init__(self, dim: int, num_heads: int = 4, attn_chunk_size: int | None = 1024) -> None:
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(dim, affine=True)
        self.attn = MultiDConvHeadTrans(dim, num_heads=num_heads, attn_chunk_size=attn_chunk_size)
        self.norm2 = nn.InstanceNorm2d(dim, affine=True)
        self.ffn = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerEncoder(nn.Module):
    """Encoder producing three scales with AdaIN prompt injection."""

    def __init__(
        self,
        channels: List[int],
        prompt_dim: int,
        use_checkpoint: bool = False,
        attn_chunk_size: int | None = 1024,
    ) -> None:
        super().__init__()
        self.embed = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
        self.stage1 = RestormerBlock(channels[0], num_heads=2, attn_chunk_size=attn_chunk_size)
        self.down1 = nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=4)
        self.stage2 = RestormerBlock(channels[1], num_heads=4, attn_chunk_size=attn_chunk_size)
        self.down2 = nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2)
        self.stage3 = RestormerBlock(channels[2], num_heads=4, attn_chunk_size=attn_chunk_size)
        self.adain1 = AdaIN(channels[0], prompt_dim)
        self.adain2 = AdaIN(channels[1], prompt_dim)
        self.adain3 = AdaIN(channels[2], prompt_dim)
        self.use_checkpoint = use_checkpoint

    def _run_block(self, block: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        _, _, height, width = x.shape
        pad_h = (4 - height % 4) % 4
        pad_w = (4 - width % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        feat1 = self._run_block(self.stage1, self.embed(x))
        feat1 = self.adain1(feat1, prompt)
        feat2 = self._run_block(self.stage2, self.down1(feat1))
        feat2 = self.adain2(feat2, prompt)
        feat3 = self._run_block(self.stage3, self.down2(feat2))
        feat3 = self.adain3(feat3, prompt)
        return feat1, feat2, feat3
