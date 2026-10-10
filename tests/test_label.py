import numpy as np
import torch
from PIL import Image

from catlandmarks.label import SheetLayout, run, sheet
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

    info = run(tmp_path / "model", tmp_path / "afhq", tmp_path / "labels.npz", dev=torch.device("cpu"))
    z = np.load(tmp_path / "labels.npz")
    assert info["images"] == 5 and z["points"].shape == (5, 9, 2) and z["confidence"].shape == (5, 9)
    assert z["ids"][0] == "train/cat/0.png"
    assert z["points"].min() >= 0 and z["points"].max() <= 63
    assert sheet(tmp_path / "labels.npz", tmp_path / "afhq", tmp_path / "sheet.png", SheetLayout(n=4, side=32)).exists()



def test_a_padded_frame_maps_points_back_to_image_pixels_even_outside_it(tmp_path, monkeypatch):
    (tmp_path / "afhq" / "train" / "cat").mkdir(parents=True)
    (tmp_path / "afhq" / "test" / "cat").mkdir(parents=True)
    Image.new("RGB", (64, 64), (200, 90, 30)).save(tmp_path / "afhq" / "train" / "cat" / "0.png")

    class Corners(torch.nn.Module):            # puts every landmark on the input's first and last pixel
        encoder = LandmarkModel(TINY).encoder

        def forward(self, x):
            p = torch.tensor([[0.0, 0.0]] * 4 + [[31.0, 31.0]] * 5).expand(len(x), 9, 2)
            return p, torch.ones(len(x), 9)
    monkeypatch.setattr("catlandmarks.label.load", lambda *_: Corners())

    run(tmp_path / "model", tmp_path / "afhq", tmp_path / "labels.npz", pad=2.0, dev=torch.device("cpu"))
    p = np.load(tmp_path / "labels.npz")["points"][0]
    # the 32-pixel input holds the 64-pixel image shrunk to 16 pixels from input pixel 8 on: 4 image pixels each
    assert np.allclose(p[0], [-30.5, -30.5]) and np.allclose(p[8], [93.5, 93.5])
