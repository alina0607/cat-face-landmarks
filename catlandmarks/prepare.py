"""Cut every face out once and store the crops as one array, so training reads pixels instead of decoding JPEGs.

    python -m catlandmarks.prepare cat --root data/cat_dataset --out data/cache/cat
    python -m catlandmarks.prepare afhq --root data/afhq --out data/cache/afhq

Writes, under --out:
    images.npy   uint8 (N, size, size, 3), memory-mappable
    points.npy   float32 (N, 9, 2), the landmarks in crop pixels (CAT only)
    inside.npy   float32 (N,), the share of each crop that lies on the photo (CAT only)
    ids.txt      one image id per line
    splits.npz   index arrays: "train", "val" and "test" for CAT; AFHQ's own "train" and "test"
    info.json    the settings and counts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .afhq import SPLITS as AFHQ_SPLITS
from .afhq import afhq_cats
from .cat_dataset import CatDataset
from .faces import crop, face_box, to_crop
from .landmarks import NUM_LANDMARKS

CONTEXT = 2.2        # a crop is this many face sides wide: room for zooming out to 2× and shifting in training
SIZE = 320           # crop side; at CONTEXT 2.2 the face spans 145 px, close to the median face's native size
SPLIT = (0.8, 0.1, 0.1)


def split_indices(n: int, seed: int = 0, fractions=SPLIT) -> dict[str, np.ndarray]:
    """A seeded random split into train / val / test, each sorted so memory-mapped reads stay sequential."""
    order = np.random.default_rng(seed).permutation(n)
    n_val, n_test = round(n * fractions[1]), round(n * fractions[2])
    parts = {"val": order[:n_val], "test": order[n_val:n_val + n_test], "train": order[n_val + n_test:]}
    return {k: np.sort(v) for k, v in parts.items()}


def prepare_cat(root: Path, out: Path, size: int = SIZE, context: float = CONTEXT, seed: int = 0) -> dict:
    data = CatDataset(root)
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (len(data), size, size, 3))
    points = np.zeros((len(data), NUM_LANDMARKS, 2), np.float32)
    inside = np.zeros(len(data), np.float32)
    start = time.time()
    for i, photo in enumerate(data.photos):
        box = face_box(photo.points, context)
        images[i], inside[i] = crop(data.image(photo), box, size)
        points[i] = to_crop(photo.points, box, size)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(data)} faces, {time.time() - start:.0f} s", flush=True)
    images.flush()
    data.close()
    np.save(out / "points.npy", points)
    np.save(out / "inside.npy", inside)
    (out / "ids.txt").write_text("\n".join(p.id for p in data.photos) + "\n")
    splits = split_indices(len(data), seed)
    np.savez(out / "splits.npz", **splits)
    info = {"source": "CAT dataset", "faces": len(data), "size": size, "context": context, "seed": seed,
            "splits": {k: len(v) for k, v in splits.items()},
            "inside_below_0.9": int((inside < 0.9).sum()), "seconds": round(time.time() - start)}
    (out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def prepare_afhq(root: Path, out: Path, size: int = SIZE) -> dict:
    """AFHQ's cats, already aligned and framed, only resized (Lanczos) to the CAT crops' size."""
    files = {split: afhq_cats(root, split) for split in AFHQ_SPLITS}
    every = [f for split in AFHQ_SPLITS for f in files[split]]
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (len(every), size, size, 3))
    start = time.time()
    for i, path in enumerate(every):
        images[i] = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(every)} faces, {time.time() - start:.0f} s", flush=True)
    images.flush()
    (out / "ids.txt").write_text("\n".join(p.relative_to(root).as_posix() for p in every) + "\n")
    bounds = np.cumsum([0] + [len(files[s]) for s in AFHQ_SPLITS])
    np.savez(out / "splits.npz", **{s: np.arange(bounds[k], bounds[k + 1]) for k, s in enumerate(AFHQ_SPLITS)})
    info = {"source": "AFHQ v2 cats", "faces": len(every), "size": size,
            "splits": {s: len(files[s]) for s in AFHQ_SPLITS}, "seconds": round(time.time() - start)}
    (out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut out face crops and store them for training.")
    parser.add_argument("dataset", choices=["cat", "afhq"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.dataset == "cat":
        info = prepare_cat(args.root, args.out, args.size, seed=args.seed)
    else:
        info = prepare_afhq(args.root, args.out, args.size)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
