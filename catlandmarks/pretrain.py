"""Masked-autoencoder pretraining on unlabeled cat faces.

The pool is the training split of both caches: CAT dataset crops (their landmarks unused here) and AFHQ cats.
Validation and test images of either dataset are never seen, so later evaluations stay clean. Each step draws
a random batch from the pool, cuts a random resized crop out of every image on the GPU, and trains the MAE.

    python -m catlandmarks.pretrain speed                       # time training steps
    python -m catlandmarks.pretrain train --out runs/mae         # resumes from runs/mae/last.pt if present
    python -m catlandmarks.pretrain show --out runs/mae          # reconstructions of held-out faces
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
from .mae import MAE, MAEConfig
from .vit import ViTConfig

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)      # ImageNet statistics, as in MAE
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class TrainConfig:
    cache: str = "data/cache"
    batch: int = 128
    epochs: int = 400
    warmup_epochs: int = 20
    base_lr: float = 1.5e-4                  # scaled by batch / 256, as in MAE
    weight_decay: float = 0.05
    crop_scale: tuple = (0.25, 1.0)
    seed: int = 0


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ImagePool:
    """Images from several memory-mapped caches, addressed as one list."""

    def __init__(self, cache: Path, splits: dict[str, str]):
        self.parts = []
        for name, split in splits.items():
            images = np.load(cache / name / "images.npy", mmap_mode="r")
            index = np.load(cache / name / "splits.npz")[split]
            self.parts.append((images, index))
        self.offsets = np.cumsum([0] + [len(i) for _, i in self.parts])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def get(self, which: np.ndarray) -> np.ndarray:
        out = []
        for k, (images, index) in enumerate(self.parts):
            local = which[(which >= self.offsets[k]) & (which < self.offsets[k + 1])] - self.offsets[k]
            if len(local):
                out.append(images[np.sort(index[local])])
        return np.concatenate(out)


def to_input(batch: np.ndarray, dev: torch.device) -> torch.Tensor:
    x = torch.from_numpy(batch).to(dev).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN.to(dev)) / STD.to(dev)


@dataclass(frozen=True)
class ViewSpec:
    """The random views MAE trains on: their side in pixels and the share of the image's area they cover."""

    size: int
    crop_scale: tuple[float, float]


def views(pool: ImagePool, which: np.ndarray, spec: ViewSpec, gen: torch.Generator,
          dev: torch.device | str) -> torch.Tensor:
    """Random resized crops (mirrored half the time) of the chosen pool images, as normalised model inputs."""
    images = to_input(pool.get(which), dev)
    return augment.apply(images, augment.random_resized_crop(len(images), spec.crop_scale, gen=gen), spec.size)


def lr_at(epoch: float, cfg: TrainConfig) -> float:
    peak = cfg.base_lr * cfg.batch / 256
    if epoch < cfg.warmup_epochs:
        return peak * epoch / cfg.warmup_epochs
    return peak * 0.5 * (1 + math.cos(math.pi * (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)))


def pools(cfg: TrainConfig) -> tuple[ImagePool, ImagePool]:
    cache = Path(cfg.cache)
    return (ImagePool(cache, {"cat": "train", "afhq": "train"}),
            ImagePool(cache, {"cat": "val", "afhq": "test"}))


@torch.no_grad()
def held_out_loss(model: MAE, pool: ImagePool, cfg: TrainConfig, dev: torch.device | str, n: int = 512) -> float:
    """Loss on a fixed set of held-out faces with fixed masks, comparable from epoch to epoch."""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    which = np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)
    size = model.cfg.encoder.image_size
    total = 0.0
    for i in range(0, len(which), cfg.batch):
        chunk = which[i:i + cfg.batch]
        images = augment.apply(to_input(pool.get(chunk), dev), augment.identity(len(chunk)), size)
        loss, _, _ = model(images, gen)
        total += loss.item() * len(images)
    model.train()
    return total / len(which)


def train(cfg: TrainConfig, out: Path, mae_cfg: MAEConfig | None = None, dev: torch.device | None = None) -> None:
    mae_cfg = mae_cfg or MAEConfig()
    dev = dev or device()
    out.mkdir(parents=True, exist_ok=True)
    train_pool, val_pool = pools(cfg)
    torch.manual_seed(cfg.seed)
    model = MAE(mae_cfg).to(dev)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "token" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "token" not in n)]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=0, betas=(0.9, 0.95))
    start = 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start = state["epoch"]
        print(f"resumed after epoch {start}", flush=True)
    else:
        config = {"train": asdict(cfg), "mae": asdict(mae_cfg), "pool": len(train_pool), "held_out": len(val_pool)}
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    steps = len(train_pool) // cfg.batch
    spec = ViewSpec(mae_cfg.encoder.image_size, cfg.crop_scale)
    for epoch in range(start, cfg.epochs):
        rng = np.random.default_rng(cfg.seed * 100_000 + epoch)             # per epoch, so a resume repeats it exactly
        gen = torch.Generator().manual_seed(cfg.seed * 100_000 + epoch)
        order, t0, running = rng.permutation(len(train_pool)), time.time(), 0.0
        model.train()
        for s in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch + s / steps, cfg)
            loss, _, _ = model(views(train_pool, order[s * cfg.batch:(s + 1) * cfg.batch], spec, gen, dev), gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        record = {"epoch": epoch + 1, "loss": round(running / steps, 4),
                  "held_out": round(held_out_loss(model, val_pool, cfg, dev), 4),
                  "lr": round(lr_at(epoch + 1, cfg), 7), "seconds": round(time.time() - t0)}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1,
                 "train": asdict(cfg), "mae": asdict(mae_cfg)}
        torch.save(state, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")
    torch.save({"encoder": model.encoder.state_dict(), "config": asdict(mae_cfg.encoder)}, out / "encoder.pt")


def speed(cfg: TrainConfig, steps: int = 20, mae_cfg: MAEConfig | None = None) -> dict[str, float]:
    mae_cfg = mae_cfg or MAEConfig()
    dev = device()
    train_pool, _ = pools(cfg)
    model = MAE(mae_cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen, rng, times = torch.Generator().manual_seed(0), np.random.default_rng(0), []
    spec = ViewSpec(mae_cfg.encoder.image_size, cfg.crop_scale)
    for s in range(steps + 3):
        t0 = time.time()
        which = rng.choice(len(train_pool), cfg.batch, replace=False)
        loss, _, _ = model(views(train_pool, which, spec, gen, dev), gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss.item()
        if s >= 3:
            times.append(time.time() - t0)
    per = float(np.mean(times))
    record = {"parameters": sum(p.numel() for p in model.parameters()), "pool": len(train_pool), "batch": cfg.batch,
              "seconds_per_step": round(per, 3),
              "minutes_per_epoch": round(per * (len(train_pool) // cfg.batch) / 60, 2)}
    if dev.type == "mps":
        record["peak_gpu_gb"] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 1)
    print(json.dumps(record), flush=True)
    return record


@torch.no_grad()
def show(out: Path, cfg: TrainConfig, n: int = 8, dev: torch.device | None = None) -> Path:
    """Held-out faces, left to right: the input, what the encoder sees, the reconstruction, and the reconstruction
    with the visible patches put back."""
    from PIL import Image
    dev = dev or device()
    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    mae_cfg = MAEConfig(encoder=ViTConfig(**state["mae"]["encoder"]),
                        **{k: v for k, v in state["mae"].items() if k != "encoder"})
    model = MAE(mae_cfg).to(dev).eval()
    model.load_state_dict(state["model"])
    _, val_pool = pools(cfg)
    which = np.linspace(0, len(val_pool) - 1, n).astype(int)
    size, patch = mae_cfg.encoder.image_size, mae_cfg.encoder.patch_size
    images = augment.apply(to_input(val_pool.get(which), dev), augment.identity(n), size)
    _, pred, mask = model(images, torch.Generator().manual_seed(0))
    recon = MAE.to_pixels(pred, images, patch)
    pixel_mask = mask.view(n, 1, size // patch, size // patch).repeat_interleave(patch, 2).repeat_interleave(patch, 3)
    seen = images * (1 - pixel_mask) + pixel_mask * ((0.5 - MEAN.to(dev)) / STD.to(dev))
    pasted = images * (1 - pixel_mask) + recon * pixel_mask
    columns = [images, seen, recon, pasted]
    rows = [torch.cat([c[i] for c in columns], 2) for i in range(n)]
    grid = (torch.cat(rows, 1)[None] * STD.to(dev) + MEAN.to(dev)).clamp(0, 1)[0]
    path = out / f"reconstructions_epoch{state['epoch']}.png"
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)
    return path


def main() -> None:
    """Command line: `speed`, `train` or `show` (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Masked-autoencoder pretraining on unlabeled cat faces.")
    parser.add_argument("stage", choices=["speed", "train", "show"])
    parser.add_argument("--out", type=Path, default=Path("runs/mae"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()
    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch:
        cfg.batch = args.batch
    if args.stage == "speed":
        speed(cfg)
    elif args.stage == "train":
        train(cfg, args.out)
    else:
        print(show(args.out, cfg), flush=True)


if __name__ == "__main__":
    main()
