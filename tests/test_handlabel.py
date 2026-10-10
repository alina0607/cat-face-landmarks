import json

import numpy as np
import pytest
from PIL import Image

from catlandmarks.handlabel import Labels, sample, score


def test_the_sample_is_fixed_and_drawn_from_the_test_split(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split / "cat").mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (8, 8)).save(tmp_path / split / "cat" / f"{i}.png")
    first = sample(tmp_path, n=3, seed=1)
    assert first == sample(tmp_path, n=3, seed=1)
    assert len(set(first)) == 3 and all(p.startswith("test/cat/") for p in first)


def test_labels_are_saved_after_every_cat_and_need_nine_points(tmp_path):
    labels = Labels(tmp_path / "hand.json")
    labels.set("test/cat/0.png", [[i, -i] for i in range(9)])
    assert Labels(tmp_path / "hand.json").points["test/cat/0.png"][8] == [8.0, -8.0]
    with pytest.raises(ValueError):
        labels.set("test/cat/1.png", [[0, 0]])


def test_score_divides_the_error_by_the_hand_placed_eye_distance(tmp_path):
    truth = np.zeros((9, 2))
    truth[1] = [100, 0]                                   # eyes 100 pixels apart
    (tmp_path / "hand.json").write_text(json.dumps({"test/cat/0.png": truth.tolist()}))
    pred = truth + [0, 5]                                 # every point 5 pixels off
    np.savez(tmp_path / "model.npz", ids=np.array(["train/cat/9.png", "test/cat/0.png"]),
             points=np.stack([np.zeros((9, 2)), pred]).astype(np.float32))
    result = score(tmp_path / "hand.json", tmp_path / "model.npz")
    assert result["cats"] == 1 and result["nme"] == pytest.approx(0.05)
    assert result["failure_rate_0.1"] == 0.0
