"""Fine-tune the landmark model on the CAT dataset, starting from the MAE encoder or from random weights.

Both starts share every setting, so the difference in their errors is what pretraining contributes. The
validation split picks the best epoch; the test split is evaluated once, at the end, by `evaluate`.

    python -m catlandmarks.finetune train --init runs/mae/encoder.pt --out runs/pose_mae
    python -m catlandmarks.finetune train --out runs/pose_scratch
    python -m catlandmarks.finetune evaluate --out runs/pose_mae --split test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import augment
from .landmarks import FLIP_PERMUTATION, NAMES
from .pose import LandmarkModel, normalised_error, parameter_groups
from .pretrain import MEAN, STD, device
from .vit import ViTConfig

EVAL_BATCH = 128


@dataclass
class FinetuneConfig:
    cache: str = "data/cache/cat"
    batch: int = 64
    epochs: int = 100
    warmup_epochs: int = 5
    base_lr: float = 1e-3                    # scaled by batch / 256, as in MAE fine-tuning
    layer_decay: float = 0.75
    weight_decay: float = 0.05
    drop_path: float = 0.1
    zoom: tuple = (1.0, 2.0)                 # training crops, in face sides
    eval_zoom: float = 1.5
    turn_deg: float = 25.0
    shift: float = 0.1
    seed: int = 0


class Faces:
    """The CAT crop cache: images, landmarks in crop pixels, splits, and the face's share of a crop."""

    def __init__(self, cache: Path):
        self.images = np.load(cache / "images.npy", mmap_mode="r")
        self.points = np.load(cache / "points.npy")
        self.splits = dict(np.load(cache / "splits.npz"))
        self.face = 1 / json.loads((cache / "info.json").read_text())["context"]
        self.size = self.images.shape[1]

    def batch(self, which: np.ndarray, dev: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Images and landmarks of the given faces, in the order asked for (read in index order, which is faster on
        a memory-mapped file, then put back)."""
        order = np.argsort(which)
        back = np.argsort(order)
        images = np.ascontiguousarray(self.images[np.asarray(which)[order]])[back]
        x = torch.from_numpy(images).to(dev).permute(0, 3, 1, 2).float() / 255
        return (x - MEAN.to(dev)) / STD.to(dev), torch.from_numpy(self.points[which]).to(dev)


def crops(faces: Faces, which: np.ndarray, theta: torch.Tensor, size: int,
          dev: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Images and landmarks through the maps; mirrored crops get their left and right landmarks swapped."""
    images, points = faces.batch(which, dev)
    out = augment.apply(images, theta, size)
    moved = augment.move_points(points, theta, faces.size, size)
    mirrored = augment.is_mirrored(theta).to(dev)
    moved = torch.where(mirrored[:, None, None], moved[:, list(FLIP_PERMUTATION)], moved)
    return out, moved


def landmark_loss(pred: torch.Tensor, true: torch.Tensor, size: int) -> torch.Tensor:
    """Mean distance in pixels over landmarks inside the crop; one cut off by the crop cannot be reached."""
    inside = ((true >= 0) & (true <= size - 1)).all(-1).float()
    return ((pred - true).norm(dim=-1) * inside).sum() / inside.sum().clamp(min=1)


def lr_scale(epoch: float, cfg: FinetuneConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return epoch / cfg.warmup_epochs
    return 0.5 * (1 + math.cos(math.pi * (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)))


def build(cfg: FinetuneConfig, init: Path | None) -> LandmarkModel:
    enc = ViTConfig()
    if init is not None:
        state = torch.load(init, map_location="cpu", weights_only=False)
        enc = ViTConfig(**state["config"])
    model = LandmarkModel(ViTConfig(**{**asdict(enc), "drop_path": cfg.drop_path}))
    if init is not None:
        model.encoder.load_state_dict(state["encoder"])
    return model


@torch.no_grad()
def errors(model: LandmarkModel, faces: Faces, split: str, cfg: FinetuneConfig,
           dev: torch.device | str) -> torch.Tensor:
    """Normalised error of every landmark of every face in a split, on the centred crop, (N, 9)."""
    model.eval()
    which, size, out = faces.splits[split], model.encoder.cfg.image_size, []
    for i in range(0, len(which), EVAL_BATCH):
        chunk = which[i:i + EVAL_BATCH]
        images, true = crops(faces, chunk, augment.centre_crop(len(chunk), faces.face, cfg.eval_zoom), size, dev)
        out.append(normalised_error(model(images)[0], true).cpu())
    model.train()
    return torch.cat(out)


def summary(err: torch.Tensor) -> dict[str, object]:
    per_face = err.mean(1)
    return {"nme": round(per_face.mean().item(), 4),
            "failure_rate_0.1": round((per_face > 0.1).float().mean().item(), 4),
            "nme_per_landmark": {n: round(v, 4) for n, v in zip(NAMES, err.mean(0).tolist(), strict=True)}}


def train(cfg: FinetuneConfig, out: Path, init: Path | None = None, dev: torch.device | None = None) -> None:
    dev = dev or device()
    out.mkdir(parents=True, exist_ok=True)
    faces = Faces(Path(cfg.cache))
    torch.manual_seed(cfg.seed)
    model = build(cfg, init).to(dev)
    groups = parameter_groups(model, cfg.base_lr * cfg.batch / 256, cfg.weight_decay, cfg.layer_decay)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    start, best = 0, float("inf")
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start, best = state["epoch"], state["best"]
        print(f"resumed after epoch {start}", flush=True)
    else:
        config = {"finetune": asdict(cfg), "init": str(init) if init else None,
                  "splits": {k: len(v) for k, v in faces.splits.items()}}
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    train_idx, size = faces.splits["train"], model.encoder.cfg.image_size
    steps = len(train_idx) // cfg.batch
    for epoch in range(start, cfg.epochs):
        rng = np.random.default_rng(cfg.seed * 100_000 + epoch)
        gen = torch.Generator().manual_seed(cfg.seed * 100_000 + epoch)
        order, t0, running = rng.permutation(train_idx), time.time(), 0.0
        for s in range(steps):
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(epoch + s / steps, cfg)
            chunk = order[s * cfg.batch:(s + 1) * cfg.batch]
            spec = augment.FaceCropSpec(cfg.zoom, cfg.turn_deg, cfg.shift)
            theta = augment.random_face_crop(len(chunk), faces.face, spec, gen)
            images, true = crops(faces, chunk, theta, size, dev)
            images = jitter(images, gen)
            loss = landmark_loss(model(images)[0], true, size)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        val = summary(errors(model, faces, "val", cfg, dev))
        record = {"epoch": epoch + 1, "train_px": round(running / steps, 3), "val_nme": val["nme"],
                  "val_failure_rate": val["failure_rate_0.1"], "seconds": round(time.time() - t0)}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if val["nme"] < best:
            best = val["nme"]
            torch.save({"model": model.state_dict(), "config": asdict(model.encoder.cfg), "epoch": epoch + 1,
                        "val": val}, out / "best.pt")
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1, "best": best}
        torch.save(state, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")


def jitter(images: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random brightness, contrast and saturation per image, on normalised inputs."""
    n, dev = images.shape[0], images.device
    u = lambda lo, hi: (lo + (hi - lo) * torch.rand(n, generator=gen)).to(dev).view(n, 1, 1, 1)  # noqa: E731
    x = images * STD.to(dev) + MEAN.to(dev)
    grey = x.mean(1, keepdim=True)
    x = grey + (x - grey) * u(0.6, 1.4)
    x = ((x - 0.5) * u(0.7, 1.3) + 0.5 + u(-0.1, 0.1)).clamp(0, 1)
    return (x - MEAN.to(dev)) / STD.to(dev)


def load(out: Path, dev: torch.device | str) -> LandmarkModel:
    """The best checkpoint of a fine-tuning run, ready for inference."""
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model = LandmarkModel(ViTConfig(**state["config"]))
    model.load_state_dict(state["model"])
    return model.to(dev).eval()


def evaluate(out: Path, split: str, cfg: FinetuneConfig, dev: torch.device | None = None) -> dict[str, object]:
    """Error statistics of a run's best checkpoint on one split, also written to <out>/eval_<split>.json."""
    dev = dev or device()
    faces = Faces(Path(cfg.cache))
    result = {"split": split, "faces": len(faces.splits[split]),
              **summary(errors(load(out, dev), faces, split, cfg, dev))}
    (out / f"eval_{split}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Command line: `train` or `evaluate` (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Fine-tune the landmark model on the CAT dataset.")
    parser.add_argument("stage", choices=["train", "evaluate"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=None, help="MAE encoder checkpoint; random weights if omitted")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    cfg = FinetuneConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.stage == "train":
        train(cfg, args.out, args.init)
    else:
        print(json.dumps(evaluate(args.out, args.split, cfg)), flush=True)


if __name__ == "__main__":
    main()
