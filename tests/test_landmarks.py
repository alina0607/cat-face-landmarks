import numpy as np
import pytest

from catlandmarks.landmarks import FLIP_PERMUTATION, NAMES, NUM_LANDMARKS, flip_points, parse_annotation


def test_parse_annotation_reads_nine_points_in_order():
    text = "9 175 160 239 162 199 199 149 121 137 78 166 93 281 101 312 96 296 133 "
    points = parse_annotation(text)
    assert points.shape == (NUM_LANDMARKS, 2)
    assert points[0].tolist() == [175, 160]
    assert points[-1].tolist() == [296, 133]


@pytest.mark.parametrize("text", ["", "8 1 2 3 4", "9 1 2 3"])
def test_parse_annotation_rejects_malformed_text(text):
    with pytest.raises(ValueError):
        parse_annotation(text)


def test_flip_permutation_swaps_each_left_name_with_its_right_name():
    for i, j in enumerate(FLIP_PERMUTATION):
        assert NAMES[j] == NAMES[i].replace("left", "#").replace("right", "left").replace("#", "right")


def test_flipping_twice_restores_the_points():
    points = np.random.default_rng(0).uniform(0, 500, (4, NUM_LANDMARKS, 2))
    np.testing.assert_allclose(flip_points(flip_points(points, 500), 500), points)


def test_flipped_eyes_stay_left_and_right_in_the_image():
    points = np.zeros((NUM_LANDMARKS, 2))
    points[0] = (100, 50)                                   # left eye
    points[1] = (200, 50)                                   # right eye
    flipped = flip_points(points, 400)
    assert flipped[0, 0] < flipped[1, 0]
    assert flipped[0].tolist() == [199, 50]                 # the old right eye, mirrored
