import torch

from catlandmarks.vit import ViT, ViTConfig, sincos_position_codes

TINY = ViTConfig(image_size=32, patch_size=8, dim=32, depth=2, heads=4)


def test_output_is_a_feature_map_of_the_patch_grid():
    model = ViT(TINY)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 32, 4, 4)


def test_position_codes_are_distinct_and_bounded():
    codes = sincos_position_codes(64, 7)
    assert codes.shape == (49, 64)
    assert codes.abs().max() <= 1
    assert torch.cdist(codes, codes).add(torch.eye(49) * 10).min() > 0.1


def test_patch_tokens_only_see_their_own_patch_before_attention():
    model = ViT(TINY)
    a = torch.zeros(1, 3, 32, 32)
    b = a.clone()
    b[..., :8, :8] = 1                                       # change the first patch only
    diff = (model.embed(a) - model.embed(b)).abs().sum(-1)[0]
    assert diff[0] > 0 and torch.all(diff[1:] == 0)


def test_drop_path_is_off_in_eval_mode():
    model = ViT(ViTConfig(image_size=32, patch_size=8, dim=32, depth=2, heads=4, drop_path=0.5)).eval()
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(model(x), model(x))
