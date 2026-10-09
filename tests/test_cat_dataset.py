import zipfile

import numpy as np
import pytest
from PIL import Image

from catlandmarks.cat_dataset import CatDataset

ANNOTATION = "9 " + " ".join(str(v) for v in range(1, 19))


def _jpeg_bytes(colour, path):
    Image.new("RGB", (32, 24), colour).save(path, format="JPEG")
    return path.read_bytes()


def test_reads_zipped_and_unpacked_photos_and_skips_unannotated_ones(tmp_path):
    zipped = _jpeg_bytes((200, 0, 0), tmp_path / "a.jpg")
    with zipfile.ZipFile(tmp_path / "CAT_DATASET_01.zip", "w") as z:
        z.writestr("CAT_00/00000001_000.jpg", zipped)
        z.writestr("CAT_00/00000001_000.jpg.cat", ANNOTATION)
        z.writestr("CAT_00/00000002_000.jpg", zipped)              # no annotation
    folder = tmp_path / "CAT_05"
    folder.mkdir()
    _jpeg_bytes((0, 0, 200), folder / "00000900_000.jpg")
    (folder / "00000900_000.jpg.cat").write_text(ANNOTATION)
    (tmp_path / "a.jpg").unlink()

    data = CatDataset(tmp_path)
    assert [p.id for p in data.photos] == ["CAT_00/00000001_000", "CAT_05/00000900_000"]
    np.testing.assert_array_equal(data[0].points[1], [3, 4])
    red, blue = (np.asarray(data.image(p)).mean((0, 1)) for p in data.photos)
    assert red[0] > 150 and blue[2] > 150
    data.close()


def test_missing_root_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CatDataset(tmp_path / "nowhere")
