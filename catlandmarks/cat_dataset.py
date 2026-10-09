"""Reading the CAT dataset (Zhang, Sun and Tang, ECCV 2008).

The dataset is distributed as folders CAT_00 ... CAT_06 of JPEG photos, each with its landmarks in a text file
of the same name plus `.cat`. The archive.org release packs the folders into CAT_DATASET_01.zip and
CAT_DATASET_02.zip; the Kaggle mirror ships them unpacked. `CatDataset` reads either, without unpacking.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .landmarks import parse_annotation

ANNOTATION_SUFFIX = ".cat"


@dataclass(frozen=True)
class CatPhoto:
    """One annotated photo. `id` is its path inside the dataset without the extension, e.g. CAT_00/00000001_000."""

    id: str
    points: np.ndarray        # (9, 2) pixel positions (x, y) in the photo
    source: Path              # the zip archive that holds it, or the photo file itself
    member: str | None        # its name inside the archive; None for an unpacked photo


class CatDataset:
    """Every annotated photo under `root`, which holds the dataset's zip archives, its folders, or both.

    Photos without an annotation are skipped. Photos are ordered by id, so indices are stable across releases.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"CAT dataset not found at {self.root}")
        found: dict[str, CatPhoto] = {}
        for archive in sorted(self.root.glob("*.zip")):
            with zipfile.ZipFile(archive) as z:
                names = set(z.namelist())
                for name in sorted(names):
                    if name.lower().endswith(".jpg") and name + ANNOTATION_SUFFIX in names:
                        points = parse_annotation(z.read(name + ANNOTATION_SUFFIX).decode())
                        found[_photo_id(name)] = CatPhoto(_photo_id(name), points, archive, name)
        for photo in sorted(self.root.rglob("*.jpg")):
            note = photo.with_name(photo.name + ANNOTATION_SUFFIX)
            if note.exists():
                name = photo.relative_to(self.root).as_posix()
                found[_photo_id(name)] = CatPhoto(_photo_id(name), parse_annotation(note.read_text()), photo, None)
        if not found:
            raise FileNotFoundError(f"no annotated photos under {self.root}")
        self.photos = [found[k] for k in sorted(found)]
        self._archives: dict[Path, zipfile.ZipFile] = {}

    def __len__(self) -> int:
        return len(self.photos)

    def __getitem__(self, index: int) -> CatPhoto:
        return self.photos[index]

    def image(self, photo: CatPhoto) -> Image.Image:
        """The photo as RGB (a few in the dataset are greyscale)."""
        if photo.member is None:
            return Image.open(photo.source).convert("RGB")
        archive = self._archives.get(photo.source)
        if archive is None:
            archive = self._archives[photo.source] = zipfile.ZipFile(photo.source)
        return Image.open(io.BytesIO(archive.read(photo.member))).convert("RGB")

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()


def _photo_id(name: str) -> str:
    """CAT_00/00000001_000 from CAT_00/00000001_000.jpg, keeping only the folder and the file name."""
    parts = Path(name).with_suffix("").parts
    return "/".join(parts[-2:])
