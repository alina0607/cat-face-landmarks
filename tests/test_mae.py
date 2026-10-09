import torch

from catlandmarks.mae import MAE, MAEConfig, patchify, random_masking, unpatchify
from catlandmarks.vit import ViTConfig

TINY = MAEConfig(encoder=ViTConfig(image_size=32, patch_size=8, dim=32, depth=2, heads=4),
                 decoder_dim=32, decoder_depth=1, decoder_heads=4)


def test_patchify_round_trips_and_orders_patches_row_by_row():
    images = torch.rand(2, 3, 32, 32)
    patches = patchify(images, 8)
    assert patches.shape == (2, 16, 192)
    torch.testing.assert_close(unpatchify(patches, 8), images)
    torch.testing.assert_close(patches[0, 1].view(8, 8, 3).permute(2, 0, 1), images[0, :, :8, 8:16])


def test_masking_removes_the_requested_share_and_marks_kept_tokens():
    kept, mask = random_masking(4, 196, 0.75, "cpu", torch.Generator().manual_seed(0))
    assert kept.shape == (4, 49)
    assert torch.all(mask.sum(1) == 147)
    assert torch.all(mask.gather(1, kept) == 0)


def test_loss_only_counts_removed_patches():
    model = MAE(TINY)
    images = torch.rand(2, 3, 32, 32)
    loss, pred, mask = model(images, torch.Generator().manual_seed(0))
    assert pred.shape == (2, 16, 192) and mask.shape == (2, 16)
    target = patchify(images, 8)
    target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
    per_patch = ((pred - target) ** 2).mean(-1)
    torch.testing.assert_close(loss, per_patch[mask.bool()].mean())


def test_a_tiny_model_learns_to_fill_in_a_fixed_image():
    torch.manual_seed(0)
    model = MAE(TINY)
    image = torch.rand(1, 3, 32, 32).repeat(8, 1, 1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss, _, _ = model(image)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first
