"""Square face crops around the landmarks, and the landmarks' positions inside them.

Landmarks follow the dataset: a landmark at (x, y) is the centre of pixel (x, y). A `Box` lives in continuous
image coordinates, in which pixel i covers [i, i + 1) and its centre is i + 0.5.

The face's extent is the landmarks' bounding box; a crop is centred on that box and is `context` times its
longer side, so `context` = 1 is a tight crop and larger values keep the surroundings that random zooms and
shifts in training will need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Box:
    """A square in continuous image coordinates (pixel i covers [i, i + 1)): its centre (x, y) and side."""

    cx: float
    cy: float
    side: float


def face_box(points: np.ndarray, context: float) -> Box:
    lo, hi = points.min(0), points.max(0)
    centre = (lo + hi) / 2 + 0.5
    return Box(float(centre[0]), float(centre[1]), float((hi - lo).max() * context))


def crop(image: Image.Image, box: Box, size: int) -> tuple[np.ndarray, float]:
    """The box resampled to size × size (bicubic), and the share of it that lies inside the image.

    Area outside the image is filled with the image's mean colour, which keeps the crop's statistics close to
    the photo's; the share lets callers skip crops that are mostly fill.
    """
    scale = box.side / size
    left, top = box.cx - box.side / 2, box.cy - box.side / 2
    # PIL maps continuous output coordinates (u, v) to input (scale·u + left, scale·v + top)
    coeffs = (scale, 0.0, left, 0.0, scale, top)
    fill = tuple(int(c) for c in np.asarray(image).reshape(-1, 3).mean(0))
    out = image.transform((size, size), Image.AFFINE, coeffs, resample=Image.BICUBIC, fillcolor=fill)
    inside = Image.new("L", image.size, 255).transform((size, size), Image.AFFINE, coeffs, fillcolor=0)
    return np.asarray(out), float(np.asarray(inside).mean() / 255)


def to_crop(points: np.ndarray, box: Box, size: int) -> np.ndarray:
    """Landmarks in image pixels → the same landmarks in the crop's pixels."""
    scale = box.side / size
    corner = np.array([box.cx - box.side / 2, box.cy - box.side / 2])
    return (points + 0.5 - corner) / scale - 0.5
