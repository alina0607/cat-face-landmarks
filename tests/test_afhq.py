import pytest
from PIL import Image

from catlandmarks.afhq import afhq_cats


def test_lists_only_the_cats_of_the_split(tmp_path):
    for split, animal, name in [("train", "cat", "b.png"), ("train", "cat", "a.png"), ("train", "dog", "c.png"),
                                ("test", "cat", "d.png")]:
        (tmp_path / split / animal).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(tmp_path / split / animal / name)
    assert [p.name for p in afhq_cats(tmp_path, "train")] == ["a.png", "b.png"]
    assert [p.name for p in afhq_cats(tmp_path, "test")] == ["d.png"]


def test_unknown_split_and_missing_folder_are_errors(tmp_path):
    with pytest.raises(ValueError):
        afhq_cats(tmp_path, "val")
    with pytest.raises(FileNotFoundError):
        afhq_cats(tmp_path, "train")
