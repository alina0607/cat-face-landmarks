"""Masked autoencoder (He et al., CVPR 2022).

A random 75% of an image's patches are removed before the encoder, which therefore sees only a quarter of the
tokens; a small decoder receives the encoded visible tokens plus a shared learned "mask" token at every removed
position and predicts the removed patches' pixels. The loss is the mean squared error on removed patches only,
against pixels normalised within each patch, which the paper found to give better representations. After
pretraining the decoder is discarded and the encoder is fine-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .vit import Block, ViT, ViTConfig, init_weights, sincos_position_codes


@dataclass(frozen=True)
class MAEConfig:
    encoder: ViTConfig = field(default_factory=ViTConfig)
    decoder_dim: int = 256
    decoder_depth: int = 4
    decoder_heads: int = 8
    mask_ratio: float = 0.75


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) → (B, patches, patch·patch·C), patches in row-major order like the encoder's tokens."""
    b, c, h, w = images.shape
    x = images.reshape(b, c, h // patch, patch, w // patch, patch)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // patch) * (w // patch), patch * patch * c)


def unpatchify(patches: torch.Tensor, patch: int, channels: int = 3) -> torch.Tensor:
    b, n, _ = patches.shape
    g = int(n ** 0.5)
    x = patches.reshape(b, g, g, patch, patch, channels).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, g * patch, g * patch)


def random_masking(n: int, tokens: int, ratio: float, device: torch.device | str,
                   gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per image, a random subset of tokens to keep: (kept indices (n, keep), mask (n, tokens), 1 = removed)."""
    keep = int(round(tokens * (1 - ratio)))
    noise = torch.rand(n, tokens, generator=gen).to(device)
    kept = noise.argsort(1)[:, :keep]
    mask = torch.ones(n, tokens, device=device)
    mask.scatter_(1, kept, 0)
    return kept, mask


class MAE(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg
        enc = cfg.encoder
        self.encoder = ViT(enc)
        self.decoder_embed = nn.Linear(enc.dim, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        self.register_buffer("decoder_pos", sincos_position_codes(cfg.decoder_dim, enc.grid)[None], persistent=False)
        self.decoder = nn.ModuleList([Block(cfg.decoder_dim, cfg.decoder_heads, 4.0)
                                      for _ in range(cfg.decoder_depth)])
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        self.predict = nn.Linear(cfg.decoder_dim, enc.patch_size ** 2 * 3)
        for m in (self.decoder_embed, self.decoder, self.decoder_norm, self.predict):
            m.apply(init_weights)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, images: torch.Tensor,
                gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(loss, predicted patches (B, tokens, patch²·3), mask (B, tokens))."""
        tokens = self.encoder.embed(images)
        b, n, d = tokens.shape
        kept, mask = random_masking(b, n, self.cfg.mask_ratio, images.device, gen)
        visible = torch.gather(tokens, 1, kept[..., None].expand(-1, -1, d))
        encoded = self.decoder_embed(self.encoder.encode(visible))
        full = self.mask_token.expand(b, n, -1).clone()
        full.scatter_(1, kept[..., None].expand(-1, -1, full.shape[-1]), encoded)
        x = full + self.decoder_pos
        for block in self.decoder:
            x = block(x)
        pred = self.predict(self.decoder_norm(x))
        target = patchify(images, self.cfg.encoder.patch_size)
        mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        loss = (((pred - target) ** 2).mean(-1) * mask).sum() / mask.sum()
        return loss, pred, mask

    @staticmethod
    def to_pixels(pred: torch.Tensor, images: torch.Tensor, patch: int) -> torch.Tensor:
        """Predicted normalised patches → pixels, using each true patch's own mean and spread (for display)."""
        target = patchify(images, patch)
        mean, std = target.mean(-1, keepdim=True), (target.var(-1, keepdim=True) + 1e-6).sqrt()
        return unpatchify(pred * std + mean, patch, images.shape[1])
