import torch
from torch import nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class MDTA(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    def __init__(self, channels: int, expansion_factor: float = 2.66):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, expansion_factor: float = 2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AdditivePromptInjection(nn.Module):
    def __init__(self, prompt_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(prompt_dim, feature_dim)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if prompt is None:
            return x
        prompt_proj = self.proj(prompt).unsqueeze(-1).unsqueeze(-1)
        return x + strength * prompt_proj


class PromptPredictor(nn.Module):
    def __init__(self, in_channels: int, pool_size: int, prompt_dim: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Linear(in_channels, pool_size)
        self.pool = nn.Parameter(torch.randn(pool_size, prompt_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.gap(x).flatten(1)
        logits = self.mlp(pooled)
        weights = F.softmax(logits, dim=-1)
        prompt = torch.matmul(weights, self.pool)
        return prompt


class LightweightDecoder(nn.Module):
    def __init__(self, dims: list[int], out_channels: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
            for dim in dims
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [head(feature) for head, feature in zip(self.heads, features)]


class PromptedRestormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_heads: list[int] | None = None,
        modality_pool_size: int = 2,
        degradation_pool_size: int = 4,
        prompt_dim: int = 64,
        modality_strong: float = 1.0,
        modality_weak: float = 0.2,
        degradation_strong: float = 1.0,
    ):
        super().__init__()
        num_blocks = num_blocks or [2, 2, 4]
        num_heads = num_heads or [1, 2, 4]

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, num_heads[2]) for _ in range(num_blocks[2])])

        self.modality_predictor = PromptPredictor(dim * 4, modality_pool_size, prompt_dim)
        self.degradation_predictor = PromptPredictor(dim, degradation_pool_size, prompt_dim)

        self.modality_inject_f2 = AdditivePromptInjection(prompt_dim, dim * 4)
        self.modality_inject_f1 = AdditivePromptInjection(prompt_dim, dim * 2)
        self.degradation_inject_f0 = AdditivePromptInjection(prompt_dim, dim)

        self.modality_strong = modality_strong
        self.modality_weak = modality_weak
        self.degradation_strong = degradation_strong

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.patch_embed(x)
        f0 = self.encoder_level1(f0)
        f1 = self.down1_2(f0)
        f1 = self.encoder_level2(f1)
        f2 = self.down2_3(f1)
        f2 = self.encoder_level3(f2)

        modality_prompt = self.modality_predictor(f2)
        degradation_prompt = self.degradation_predictor(f0)

        f2 = self.modality_inject_f2(f2, modality_prompt, strength=self.modality_strong)
        f1 = self.modality_inject_f1(f1, modality_prompt, strength=self.modality_weak)
        f0 = self.degradation_inject_f0(f0, degradation_prompt, strength=self.degradation_strong)

        return f0, f1, f2


class Stage1PromptedRestormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_heads: list[int] | None = None,
        modality_pool_size: int = 2,
        degradation_pool_size: int = 4,
        prompt_dim: int = 64,
        modality_strong: float = 1.0,
        modality_weak: float = 0.2,
        degradation_strong: float = 1.0,
    ):
        super().__init__()
        self.encoder = PromptedRestormerEncoder(
            in_channels=in_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            modality_pool_size=modality_pool_size,
            degradation_pool_size=degradation_pool_size,
            prompt_dim=prompt_dim,
            modality_strong=modality_strong,
            modality_weak=modality_weak,
            degradation_strong=degradation_strong,
        )
        self.decoder = LightweightDecoder([dim, dim * 2, dim * 4], out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f0, f1, f2 = self.encoder(x)
        recon_full, recon_half, recon_quarter = self.decoder([f0, f1, f2])
        return {
            "features": (f0, f1, f2),
            "recon_full": recon_full,
            "recon_half": recon_half,
            "recon_quarter": recon_quarter,
        }
