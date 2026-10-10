"""Landmarks from a Vision Transformer: ViTPose's simple decoder and integral regression.

The encoder's patch features (stride 16) are upsampled by two transposed convolutions to stride 4 and turned
into one heat map per landmark by a 1×1 convolution (Xu et al., 2022, "classic decoder"). Instead of taking each
heat map's peak, the map is normalised with a softmax and the landmark is its expected position (Sun et al.,
2018): differentiable, sub-pixel, and trained directly on the distance to the true point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .landmarks import NUM_LANDMARKS
from .vit import ViT, ViTConfig


class Decoder(nn.Module):
    def __init__(self, dim: int, channels: int = 256, landmarks: int = NUM_LANDMARKS):
        super().__init__()
        layers, c = [], dim
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, channels, 4, 2, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()]
            c = channels
        self.up = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, landmarks, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.up(features))


class LandmarkModel(nn.Module):
    def __init__(self, cfg: ViTConfig, decoder_channels: int = 256):
        super().__init__()
        self.encoder = ViT(cfg)
        self.decoder = Decoder(cfg.dim, decoder_channels)
        self.stride = cfg.patch_size // 4

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Images (B, 3, S, S) → (landmarks (B, K, 2) in input pixels, each heat map's peak probability)."""
        return soft_argmax(self.decoder(self.encoder(images)), self.stride)


def soft_argmax(maps: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Heat-map logits (B, K, h, w) → expected positions in input pixels (B, K, 2), and each map's largest
    probability, which is high when the model is sure of one place and low when it hesitates."""
    b, k, h, w = maps.shape
    p = maps.flatten(2).softmax(-1).view(b, k, h, w)
    xs = (torch.arange(w, device=maps.device, dtype=p.dtype) + 0.5) * stride - 0.5   # cell centres, input pixels
    ys = (torch.arange(h, device=maps.device, dtype=p.dtype) + 0.5) * stride - 0.5
    points = torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)
    return points, p.flatten(2).amax(-1)


def normalised_error(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Each landmark's error divided by the distance between the true eyes, (B, K)."""
    eyes = (true[:, 0] - true[:, 1]).norm(dim=-1, keepdim=True)
    return (pred - true).norm(dim=-1) / eyes


def parameter_groups(model: LandmarkModel, lr: float, weight_decay: float,
                     layer_decay: float) -> list[dict[str, object]]:
    """AdamW groups with layer-wise learning-rate decay (Clark et al., 2020; used by MAE and ViTPose fine-tuning):
    the decoder gets `lr`, the last encoder block `lr·layer_decay`, each block below another factor, and the patch
    embedding the smallest. Norms and biases are not decayed."""
    depth = len(model.encoder.blocks)

    def layer(name: str) -> int:
        if name.startswith("encoder.patch_embed"):
            return 0
        if name.startswith("encoder.blocks."):
            return int(name.split(".")[2]) + 1
        return depth + 1                                              # final norm and decoder

    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        i = layer(name)
        decay = p.ndim >= 2
        key = (i, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * layer_decay ** (depth + 1 - i),
                           "weight_decay": weight_decay if decay else 0.0}
        groups[key]["params"].append(p)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return list(groups.values())
