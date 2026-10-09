"""The nine landmarks and how they behave under a horizontal flip.

The CAT dataset annotates each face with nine points in a fixed order: the two eyes, the mouth, then three
points per ear going round the head from the left ear's outer base, over its tip, to its inner base, and on
across the right ear to its outer base. "Left" and "right" are as seen in the image, not from the cat's
point of view.
"""

from __future__ import annotations

import numpy as np

NAMES = (
    "left_eye",
    "right_eye",
    "mouth",
    "left_ear_outer",
    "left_ear_tip",
    "left_ear_inner",
    "right_ear_inner",
    "right_ear_tip",
    "right_ear_outer",
)
NUM_LANDMARKS = len(NAMES)

# A mirrored face swaps left and right: index i of a flipped face holds what was FLIP_PERMUTATION[i].
FLIP_PERMUTATION = (1, 0, 2, 8, 7, 6, 5, 4, 3)


def parse_annotation(text: str) -> np.ndarray:
    """A `.cat` annotation ("9 x1 y1 x2 y2 ... x9 y9") as a (9, 2) float array of (x, y) pixel positions."""
    values = text.split()
    if not values:
        raise ValueError("empty annotation")
    count = int(values[0])
    if count != NUM_LANDMARKS or len(values) < 1 + 2 * count:
        raise ValueError(f"expected {NUM_LANDMARKS} points, got {values[:1 + 2 * NUM_LANDMARKS]}")
    return np.array(values[1:1 + 2 * count], dtype=np.float64).reshape(count, 2)


def flip_points(points: np.ndarray, width: float) -> np.ndarray:
    """Landmarks of an image of the given width mirrored left to right, renamed so they keep their meaning.

    Pixel centres sit at integer coordinates, so x maps to width - 1 - x.
    """
    flipped = points.copy()
    flipped[..., 0] = width - 1 - flipped[..., 0]
    return flipped[..., FLIP_PERMUTATION, :]
