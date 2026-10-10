import numpy as np
import torch
from PIL import Image

from catlandmarks.label import run, sheet
from catlandmarks.pose import LandmarkModel
from catlandmarks.vit import ViTConfig

TINY = ViTConfig(image_size=32, patch_size=8, dim=32, depth=1, heads=4)


def test_labels_every_cat_in_the_source_images_pixels(tmp_path):
    for split, n in [("train", 3), ("test", 2)]:
        (tmp_path / "afhq" / split / "cat").mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (64, 64), (i * 50, 90, 30)).save(tmp_path / "afhq" / split / "cat" / f"{i}.png")
    model = LandmarkModel(TINY)
    (tmp_path / "model").mkdir()
    torch.save({"model": model.state_dict(), "config": TINY.__dict__}, tmp_path / "model" / "best.pt")

    info = run(tmp_path / "model", tmp_path / "afhq", tmp_path / "labels.npz", bs=2, dev=torch.device("cpu"))
    z = np.load(tmp_path / "labels.npz")
    assert info["images"] == 5 and z["points"].shape == (5, 9, 2) and z["confidence"].shape == (5, 9)
    assert z["ids"][0] == "train/cat/0.png"
    assert z["points"].min() >= 0 and z["points"].max() <= 63
    assert sheet(tmp_path / "labels.npz", tmp_path / "afhq", tmp_path / "sheet.png", n=4, side=32).exists()
