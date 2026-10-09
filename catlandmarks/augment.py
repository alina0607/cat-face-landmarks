"""Batched geometric augmentation on the GPU.

Every crop is an affine map from the output image to the input image, expressed in the normalised coordinates
of `torch.nn.functional.affine_grid` (-1 and 1 are the outer edges of the corner pixels, align_corners=False).
Building the maps for a whole batch and resampling once with `grid_sample` keeps augmentation off the CPU,
and the same maps move landmarks exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _uniform(n: int, lo: float, hi: float, gen: torch.Generator | None) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, generator=gen)


def random_resized_crop(n: int, scale=(0.25, 1.0), ratio=(3 / 4, 4 / 3), flip: bool = True,
                        gen: torch.Generator | None = None) -> torch.Tensor:
    """(n, 2, 3) maps for crops covering `scale` of the image's area with aspect ratio in `ratio` (log-uniform),
    placed uniformly inside the image, mirrored with probability ½ (Szegedy et al., 2015; as in MAE)."""
    area = _uniform(n, *scale, gen)
    log_ratio = _uniform(n, math.log(ratio[0]), math.log(ratio[1]), gen)
    w = (area * log_ratio.exp()).sqrt().clamp(max=1)            # half-widths in normalised units are w, h
    h = (area / log_ratio.exp()).sqrt().clamp(max=1)
    cx = _uniform(n, -1, 1, gen) * (1 - w)
    cy = _uniform(n, -1, 1, gen) * (1 - h)
    sx = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0) if flip else torch.ones(n)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 2] = w * sx, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    return theta


def identity(n: int) -> torch.Tensor:
    """(n, 2, 3) maps that keep the whole image: `apply` with them only resizes."""
    return torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).repeat(n, 1, 1)


def apply(images: torch.Tensor, theta: torch.Tensor, size: int) -> torch.Tensor:
    """Resample (B, C, H, W) images through the maps into (B, C, size, size), bilinear, edges repeated."""
    grid = F.affine_grid(theta.to(images.device, images.dtype), (images.shape[0], images.shape[1], size, size),
                         align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def to_normalised(points: torch.Tensor, size: int) -> torch.Tensor:
    """Pixel positions (pixel centres at integers) in a size × size image → affine_grid coordinates."""
    return (points + 0.5) / size * 2 - 1


def to_pixels(points: torch.Tensor, size: int) -> torch.Tensor:
    return (points + 1) / 2 * size - 0.5


def move_points(points: torch.Tensor, theta: torch.Tensor, in_size: int, out_size: int) -> torch.Tensor:
    """Landmarks (B, K, 2) in input pixels → output pixels of the crops made by `apply` with the same maps.

    A map sends output coordinates o to input coordinates A·o + t, so a point p lands at A⁻¹(p - t).
    """
    a, t = theta[:, :, :2].to(points), theta[:, :, 2].to(points)
    p = to_normalised(points, in_size) - t[:, None]
    return to_pixels(torch.linalg.solve(a[:, None], p[..., None])[..., 0], out_size)
