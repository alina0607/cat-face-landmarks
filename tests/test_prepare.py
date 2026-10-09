import json
import zipfile

import numpy as np
from PIL import Image

from catlandmarks.prepare import prepare_afhq, prepare_cat, split_indices


def test_split_is_a_seeded_partition():
    a, b = split_indices(1000, seed=3), split_indices(1000, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    joined = np.concatenate(list(a.values()))
    assert sorted(joined.tolist()) == list(range(1000))
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (800, 100, 100)


def test_prepare_writes_crops_with_landmarks_on_the_face(tmp_path):
    image = np.zeros((120, 160, 3), np.uint8)
    points = np.array([[60, 60], [100, 60], [80, 85], [50, 40], [52, 20], [65, 35],
                       [95, 35], [108, 20], [110, 40]])
    for x, y in points:
        image[y - 1:y + 2, x - 1:x + 2] = 255
    Image.fromarray(image).save(tmp_path / "p.png")
    with zipfile.ZipFile(tmp_path / "CAT_DATASET_01.zip", "w") as z:
        z.write(tmp_path / "p.png", "CAT_00/00000001_000.jpg")     # PNG bytes under a .jpg name decode fine
        z.writestr("CAT_00/00000001_000.jpg.cat", "9 " + " ".join(str(v) for v in points.ravel()))
    (tmp_path / "p.png").unlink()

    info = prepare_cat(tmp_path, tmp_path / "cache", size=64, context=1.5)
    images = np.load(tmp_path / "cache" / "images.npy")
    crop_points = np.load(tmp_path / "cache" / "points.npy")
    assert images.shape == (1, 64, 64, 3) and crop_points.shape == (1, 9, 2)
    grey = images[0].mean(-1)
    for x, y in np.round(crop_points[0]).astype(int):
        assert grey[y, x] > 60                                     # each landmark sits on its dot
    assert info["faces"] == 1
    assert json.loads((tmp_path / "cache" / "info.json").read_text())["context"] == 1.5


def test_prepare_afhq_keeps_the_official_split_and_resizes(tmp_path):
    for split, n in [("train", 3), ("test", 2)]:
        (tmp_path / split / "cat").mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (512, 512), (40 * i, 0, 0)).save(tmp_path / split / "cat" / f"{split}_{i}.png")
    info = prepare_afhq(tmp_path, tmp_path / "cache", size=32)
    splits = np.load(tmp_path / "cache" / "splits.npz")
    ids = (tmp_path / "cache" / "ids.txt").read_text().split()
    assert np.load(tmp_path / "cache" / "images.npy").shape == (5, 32, 32, 3)
    assert [ids[i] for i in splits["test"]] == ["test/cat/test_0.png", "test/cat/test_1.png"]
    assert info["splits"] == {"train": 3, "test": 2}
