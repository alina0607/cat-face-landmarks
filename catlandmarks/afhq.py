"""Reading the cat faces of AFHQ (Choi et al., StarGAN v2, CVPR 2020).

AFHQ v2 unpacks to train/{cat,dog,wild} and test/{cat,dog,wild}, 512×512 PNGs of aligned animal faces with no
landmarks. Only the cats are used here: as unlabeled images for self-supervised pretraining, and, for a small
hand-labeled sample of the test split, to measure how well landmarks learned on the CAT dataset transfer.
"""

from __future__ import annotations

from pathlib import Path

SPLITS = ("train", "test")
SUFFIXES = (".png", ".jpg")


def afhq_cats(root: str | Path, split: str) -> list[Path]:
    """The cat images of one split, sorted by name."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    folder = Path(root) / split / "cat"
    if not folder.is_dir():
        raise FileNotFoundError(f"AFHQ cats not found at {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES)
