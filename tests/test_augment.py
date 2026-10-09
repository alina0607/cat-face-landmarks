import torch

from catlandmarks.augment import apply, move_points, random_resized_crop


def test_identity_map_resizes_the_whole_image():
    images = torch.rand(1, 3, 16, 16)
    theta = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    torch.testing.assert_close(apply(images, theta, 16), images)


def test_crops_stay_inside_the_image_and_cover_the_requested_area():
    theta = random_resized_crop(2000, scale=(0.3, 0.6), gen=torch.Generator().manual_seed(0))
    w, h, cx, cy = theta[:, 0, 0].abs(), theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2]
    assert torch.all(cx.abs() + w <= 1 + 1e-6) and torch.all(cy.abs() + h <= 1 + 1e-6)
    area = w * h
    assert area.min() >= 0.3 - 1e-6 and area.max() <= 0.6 + 1e-6
    assert 0.4 < (theta[:, 0, 0] < 0).float().mean() < 0.6          # about half are mirrored


def test_moved_points_follow_the_pixels_they_mark():
    gen = torch.Generator().manual_seed(1)
    images = torch.zeros(8, 1, 64, 64)
    points = torch.randint(20, 44, (8, 1, 2)).float()
    for i, (x, y) in enumerate(points[:, 0].long()):
        images[i, 0, y - 1:y + 2, x - 1:x + 2] = 1
    theta = random_resized_crop(8, scale=(0.5, 0.9), gen=gen)
    out = apply(images, theta, 48)
    moved = move_points(points, theta, 64, 48)[:, 0]
    ys, xs = torch.meshgrid(torch.arange(48.0), torch.arange(48.0), indexing="ij")
    for i in range(8):
        m = out[i, 0]
        if m.sum() < 1:                                             # the dot fell outside this crop
            continue
        found = torch.stack([(xs * m).sum() / m.sum(), (ys * m).sum() / m.sum()])
        torch.testing.assert_close(found, moved[i], atol=0.2, rtol=0)
