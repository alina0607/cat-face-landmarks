import json

import numpy as np
import torch

from catlandmarks.mae import MAEConfig
from catlandmarks.pretrain import ImagePool, TrainConfig, show, train
from catlandmarks.vit import ViTConfig

TINY = MAEConfig(encoder=ViTConfig(image_size=32, patch_size=8, dim=32, depth=1, heads=4),
                 decoder_dim=32, decoder_depth=1, decoder_heads=4)


def _cache(root, name, n, splits):
    (root / name).mkdir(parents=True)
    rng = np.random.default_rng(len(name))
    np.save(root / name / "images.npy", rng.integers(0, 255, (n, 40, 40, 3), dtype=np.uint8))
    np.savez(root / name / "splits.npz", **splits)


def _caches(root):
    _cache(root, "cat", 10, {"train": np.arange(6), "val": np.arange(6, 8), "test": np.arange(8, 10)})
    _cache(root, "afhq", 6, {"train": np.arange(4), "test": np.arange(4, 6)})


def test_pool_reads_only_the_requested_splits(tmp_path):
    _caches(tmp_path)
    pool = ImagePool(tmp_path, {"cat": "val", "afhq": "test"})
    assert len(pool) == 4
    cat = np.load(tmp_path / "cat" / "images.npy")
    np.testing.assert_array_equal(pool.get(np.array([0, 1]))[0], cat[6])


def test_train_writes_metrics_resumes_and_shows(tmp_path):
    _caches(tmp_path)
    cfg = TrainConfig(cache=str(tmp_path), batch=4, epochs=2, warmup_epochs=1)
    out = tmp_path / "run"
    train(cfg, out, TINY, torch.device("cpu"))
    cfg.epochs = 3
    train(cfg, out, TINY, torch.device("cpu"))                    # picks up after epoch 2
    epochs = [json.loads(line)["epoch"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert epochs == [1, 2, 3]
    assert (out / "encoder.pt").exists()
    assert show(out, cfg, n=2, dev=torch.device("cpu")).exists()
