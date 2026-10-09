import numpy as np
import pytest
from PIL import Image

from catlandmarks.faces import Box, crop, face_box, to_crop


def test_face_box_is_centred_on_the_landmarks_and_scaled_by_context():
    points = np.array([[10.0, 20.0], [50.0, 30.0], [30.0, 60.0]])
    box = face_box(points, context=2.0)
    assert (box.cx, box.cy, box.side) == (30.5, 40.5, 80.0)


@pytest.mark.parametrize("box,size", [(Box(60, 45, 50), 100), (Box(70.3, 52.8, 90.5), 64), (Box(40, 30, 33), 33)])
@pytest.mark.parametrize("offset", [(-0.2, -0.1), (0.15, 0.25)])
def test_a_marked_pixel_lands_where_to_crop_puts_it(box, size, offset):
    mark = np.array([[round(box.cx + offset[0] * box.side), round(box.cy + offset[1] * box.side)]])
    image = np.zeros((90, 120, 3), np.uint8)
    x, y = mark[0]
    image[y - 1:y + 2, x - 1:x + 2] = 255                   # a 3×3 dot, centred on the marked pixel
    out, _ = crop(Image.fromarray(image), box, size)
    grey = out.mean(-1).clip(0)
    ys, xs = np.mgrid[:size, :size]
    found = ((xs * grey).sum() / grey.sum(), (ys * grey).sum() / grey.sum())
    expected = to_crop(mark.astype(float), box, size)[0]
    assert found == pytest.approx(tuple(expected), abs=0.1 * max(1.0, size / box.side))


def test_inside_share_counts_the_part_of_the_box_on_the_image():
    image = Image.new("RGB", (100, 100), (10, 20, 30))
    _, whole = crop(image, Box(50, 50, 100), 50)
    _, half = crop(image, Box(100, 50, 100), 50)
    assert whole == pytest.approx(1.0)
    assert half == pytest.approx(0.5, abs=0.03)
