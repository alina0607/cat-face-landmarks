import torch

from catlandmarks.pose import LandmarkModel, normalised_error, parameter_groups, soft_argmax
from catlandmarks.vit import ViTConfig

TINY = ViTConfig(image_size=32, patch_size=8, dim=32, depth=3, heads=4)


def test_predicts_nine_points_inside_the_image():
    points, confidence = LandmarkModel(TINY, decoder_channels=16)(torch.randn(2, 3, 32, 32))
    assert points.shape == (2, 9, 2) and confidence.shape == (2, 9)
    assert points.min() >= 0 and points.max() <= 31


def test_soft_argmax_finds_a_sharp_peak_at_its_cell_centre():
    maps = torch.full((1, 1, 8, 8), -50.0)
    maps[0, 0, 2, 5] = 50.0
    points, confidence = soft_argmax(maps, stride=4)
    torch.testing.assert_close(points[0, 0], torch.tensor([5 * 4 + 1.5, 2 * 4 + 1.5]))
    assert confidence.item() > 0.99


def test_soft_argmax_lands_between_two_equal_peaks():
    maps = torch.full((1, 1, 8, 8), -50.0)
    maps[0, 0, 3, 2] = maps[0, 0, 3, 3] = 50.0
    points, confidence = soft_argmax(maps, stride=1)
    torch.testing.assert_close(points[0, 0], torch.tensor([2.5, 3.0]))
    assert abs(confidence.item() - 0.5) < 1e-3


def test_error_is_relative_to_the_eye_distance():
    true = torch.zeros(1, 9, 2)
    true[0, 1] = torch.tensor([40.0, 0.0])
    pred = true.clone()
    pred[0, 2] += torch.tensor([3.0, 4.0])
    torch.testing.assert_close(normalised_error(pred, true)[0, 2], torch.tensor(5 / 40))


def test_layer_decay_lowers_the_learning_rate_towards_the_input():
    model = LandmarkModel(TINY, decoder_channels=16)
    groups = parameter_groups(model, lr=1.0, weight_decay=0.1, layer_decay=0.5)
    by_param = {id(p): g for g in groups for p in g["params"]}
    lr = {name: by_param[id(p)]["lr"] for name, p in model.named_parameters()}
    assert lr["decoder.head.weight"] == 1.0
    assert lr["encoder.blocks.2.attn.qkv.weight"] == 0.5
    assert lr["encoder.blocks.0.attn.qkv.weight"] == 0.125
    assert lr["encoder.patch_embed.weight"] == 0.0625
    assert by_param[id(model.decoder.head.bias)]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
