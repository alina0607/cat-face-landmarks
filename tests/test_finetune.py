import json

import numpy as np
import torch

from catlandmarks import augment
from catlandmarks.finetune import Faces, FinetuneConfig, crops, evaluate, landmark_loss, train
from catlandmarks.landmarks import FLIP_PERMUTATION
from catlandmarks.pretrain import MEAN, STD
from catlandmarks.vit import ViT, ViTConfig

TINY = ViTConfig(image_size=32, patch_size=8, dim=32, depth=2, heads=4)


def _cache(root, n=12, size=48):
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(root / "images.npy", rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8))
    base = np.array([[18, 24], [30, 24], [24, 32], [14, 16], [15, 8], [20, 14], [28, 14], [33, 8], [34, 16]], float)
    np.save(root / "points.npy", (base + rng.normal(0, 1, (n, 9, 2))).astype(np.float32))
    np.savez(root / "splits.npz", train=np.arange(8), val=np.arange(8, 10), test=np.arange(10, 12))
    (root / "info.json").write_text(json.dumps({"context": 2.0}))


def test_mirrored_crops_keep_the_left_eye_on_the_left(tmp_path):
    _cache(tmp_path / "cat")
    faces = Faces(tmp_path / "cat")
    theta = augment.centre_crop(2, faces.face, 1.5)
    theta[1, 0, 0] *= -1                                             # mirror the second crop
    _, points = crops(faces, np.array([0, 0]), theta, 32, "cpu")
    assert points[0, 0, 0] < points[0, 1, 0] and points[1, 0, 0] < points[1, 1, 0]
    torch.testing.assert_close(points[1, :, 0], 31 - points[0, list(FLIP_PERMUTATION), 0])


def test_batches_come_back_in_the_order_asked_for(tmp_path):
    _cache(tmp_path / "cat")
    faces = Faces(tmp_path / "cat")
    images, points = faces.batch(np.array([5, 1, 3]), "cpu")
    raw = np.load(tmp_path / "cat" / "images.npy")
    for k, i in enumerate([5, 1, 3]):
        expected = (torch.from_numpy(raw[i]).permute(2, 0, 1).float() / 255 - MEAN[0]) / STD[0]
        torch.testing.assert_close(images[k], expected)
        torch.testing.assert_close(points[k], torch.from_numpy(faces.points[i]))


def test_landmarks_cut_off_by_the_crop_do_not_count():
    true = torch.tensor([[[5.0, 5.0], [40.0, 5.0]]])
    pred = torch.tensor([[[8.0, 9.0], [0.0, 0.0]]])
    torch.testing.assert_close(landmark_loss(pred, true, 32), torch.tensor(5.0))


def test_train_from_an_encoder_then_evaluate(tmp_path):
    _cache(tmp_path / "cat")
    torch.save({"encoder": ViT(TINY).state_dict(), "config": TINY.__dict__}, tmp_path / "encoder.pt")
    cfg = FinetuneConfig(cache=str(tmp_path / "cat"), batch=4, epochs=2, warmup_epochs=1)
    out = tmp_path / "run"
    train(cfg, out, tmp_path / "encoder.pt", torch.device("cpu"))
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2]
    result = evaluate(out, "test", cfg, torch.device("cpu"))
    assert result["faces"] == 2 and set(result["nme_per_landmark"]) >= {"left_eye", "mouth"}
