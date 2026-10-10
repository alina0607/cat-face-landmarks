"""Put the nine landmarks on every AFHQ cat, and draw a sample of them for inspection.

AFHQ faces fill their 512×512 frame, the tightest framing the model was trained on (one to two face sides). With
`--pad` the image is shrunk into the middle of a grey frame, so the face covers 1/pad of the input as it did during
training (1.5 at evaluation); the predictions are mapped back to the image's pixels. The frame also leaves room for
ear tips that AFHQ's tight crops cut off: they are placed outside the image instead of being pulled into its corners.

    python -m catlandmarks.label run --model runs/pose_mae --pad 1.5 --out data/labels/afhq_cats.npz
    python -m catlandmarks.label sheet --labels data/labels/afhq_cats.npz --afhq data/afhq --out runs/afhq_sheet.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .afhq import SPLITS, afhq_cats
from .finetune import load
from .pretrain import MEAN, STD, device

BATCH = 64
COLOURS = [(230, 40, 40), (250, 200, 0), (40, 200, 60), (40, 120, 255), (0, 210, 255), (180, 60, 255),
           (255, 60, 170), (255, 140, 0), (255, 255, 255)]


@dataclass(frozen=True)
class SheetLayout:
    """Which labeled cats a contact sheet shows and how big: n random ones (seeded), side pixels each, 8 a row."""

    n: int = 32
    seed: int = 0
    side: int = 224


@torch.no_grad()
def run(model_dir: Path, afhq: Path, out: Path, pad: float = 1.0, dev: torch.device | None = None) -> dict[str, object]:
    """Predict the landmarks of every AFHQ cat with a fine-tuned model; write them (in the images' pixels) and each
    landmark's confidence to `out`, a summary next to it, and return the summary."""
    dev = dev or device()
    model = load(model_dir, dev)
    size = model.encoder.cfg.image_size
    files = [f for split in SPLITS for f in afhq_cats(afhq, split)]
    points, confidence, source = [], [], None
    for i in range(0, len(files), BATCH):
        images = [Image.open(f).convert("RGB") for f in files[i:i + BATCH]]
        source = images[0].size[0]
        x = torch.from_numpy(np.stack([np.asarray(im.resize((size, size), Image.LANCZOS)) for im in images]))
        x = (x.to(dev).permute(0, 3, 1, 2).float() / 255 - MEAN.to(dev)) / STD.to(dev)
        inner = round(size / pad)
        framed = torch.zeros_like(x)                     # zero is the mean colour once normalised
        at = (size - inner) // 2
        framed[:, :, at:at + inner, at:at + inner] = F.interpolate(x, size=inner, mode="bilinear", antialias=True)
        p, c = model(framed)
        points.append(((p + 0.5 - at) * (source / inner) - 0.5).cpu().numpy())
        confidence.append(c.cpu().numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array([f.relative_to(afhq).as_posix() for f in files])
    np.savez(out, ids=ids, points=np.concatenate(points).astype(np.float32),
             confidence=np.concatenate(confidence).astype(np.float32))
    info = {"images": len(files), "model": str(model_dir), "image_size": source, "pad": pad}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def sheet(labels: Path, afhq: Path, out: Path, layout: SheetLayout | None = None) -> Path:
    """A grid of randomly chosen labeled cats with their landmarks drawn on."""
    layout = layout or SheetLayout()
    side = layout.side
    z = np.load(labels)
    pick = np.random.default_rng(layout.seed).choice(len(z["ids"]), min(layout.n, len(z["ids"])), replace=False)
    cols = 8
    grid = Image.new("RGB", (cols * side, -(-len(pick) // cols) * side), "white")
    for j, i in enumerate(pick):
        image = Image.open(afhq / z["ids"][i]).convert("RGB")
        scale = side / image.size[0]
        image = image.resize((side, side), Image.LANCZOS)
        draw = ImageDraw.Draw(image)
        for (x, y), colour in zip((z["points"][i] + 0.5) * scale - 0.5, COLOURS, strict=True):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour, outline=(0, 0, 0))
        grid.paste(image, ((j % cols) * side, (j // cols) * side))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    return out


def main() -> None:
    """Command line: `run` labels every cat, `sheet` draws a sample of the labels."""
    parser = argparse.ArgumentParser(description="Label AFHQ's cats with the nine landmarks.")
    parser.add_argument("stage", choices=["run", "sheet"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--afhq", type=Path, default=Path("data/afhq"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels/afhq_cats.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad", type=float, default=1.0, help="input side over face side (1.5 as in evaluation)")
    args = parser.parse_args()
    if args.stage == "run":
        print(json.dumps(run(args.model, args.afhq, args.out, args.pad)), flush=True)
    else:
        print(sheet(args.labels, args.afhq, args.out, SheetLayout(seed=args.seed)), flush=True)


if __name__ == "__main__":
    main()
