"""A plain Vision Transformer encoder (Dosovitskiy et al., ICLR 2021), as used by MAE and ViTPose.

The image is cut into non-overlapping patches, each patch is linearly embedded, a fixed 2-D sine-cosine
position code is added, and the tokens pass through pre-norm transformer blocks. There is no class token: the
encoder is only ever used for its patch tokens, which MAE reconstructs from and ViTPose decodes into heat maps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    dim: int = 384               # ViT-Small
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    drop_path: float = 0.0

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size


def sincos_position_codes(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine position codes, (grid², dim): half the channels encode the row, half the column."""
    if dim % 4:
        raise ValueError("dim must be a multiple of 4")
    freqs = 1.0 / 10000 ** (np.arange(dim // 4) / (dim // 4))
    rows, cols = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")

    def encode(pos):
        angles = pos.reshape(-1, 1) * freqs[None]
        return np.concatenate([np.sin(angles), np.cos(angles)], 1)

    return torch.from_numpy(np.concatenate([encode(rows), encode(cols)], 1)).float()


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    """Pre-norm transformer block with stochastic depth on both residual branches."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = drop_path

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path).to(x.dtype)
        return x * keep / (1 - self.drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._drop(self.attn(self.norm1(x)))
        return x + self._drop(self.mlp(self.norm2(x)))


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, cfg.dim, cfg.patch_size, cfg.patch_size)
        self.register_buffer("pos", sincos_position_codes(cfg.dim, cfg.grid)[None], persistent=False)
        rates = np.linspace(0, cfg.drop_path, cfg.depth)          # deeper blocks are dropped more often
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.heads, cfg.mlp_ratio, float(r)) for r in rates])
        self.norm = nn.LayerNorm(cfg.dim)
        self.apply(init_weights)
        # as in MAE, the patch embedding is initialised like the linear layer it is
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → patch tokens with position codes, (B, grid², dim)."""
        return self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, dim, grid, grid), the patch features laid out as an image."""
        tokens = self.encode(self.embed(images))
        g = self.cfg.grid
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, g, g)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
