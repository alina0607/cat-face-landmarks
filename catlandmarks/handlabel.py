"""Hand-label a sample of AFHQ cats in the browser, and score the model's AFHQ labels against them.

AFHQ has no landmarks, so how well the model transfers to it can only be measured on faces labeled by hand. The
sample is drawn from AFHQ's test split, which neither pretraining nor fine-tuning has seen. Each image is shown
inside a grey margin, so the tip of an ear that the crop cuts off can still be placed where it would be.

    python -m catlandmarks.handlabel serve --out data/labels/afhq_hand.json         # then open http://127.0.0.1:8765
    python -m catlandmarks.handlabel score --hand data/labels/afhq_hand.json --labels data/labels/afhq_cats.npz
"""

from __future__ import annotations

import argparse
import io
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .afhq import afhq_cats
from .label import COLOURS
from .landmarks import NAMES

PAGE = Path(__file__).with_name("handlabel.html")
MARGIN = 128                          # grey border around each 512-pixel image, in image pixels
SAMPLE = 50
DOT = 6                               # radius of a drawn landmark, in guide pixels


def sample(afhq: Path, n: int = SAMPLE, seed: int = 0) -> list[str]:
    """A fixed random sample of AFHQ test cats, as paths relative to the AFHQ folder."""
    files = afhq_cats(afhq, "test")
    return [f.relative_to(afhq).as_posix() for f in random.Random(seed).sample(files, min(n, len(files)))]


def reference(cache: Path, index: int = 5489, side: int = 320) -> bytes:
    """A labeled CAT-dataset face (by default a frontal tabby with both ears clear), cut to the face, with its
    points numbered in order, as a PNG: the guide for where each point goes."""
    points = np.load(cache / "points.npy")[index]
    centre, half = (points.min(0) + points.max(0)) / 2, np.ptp(points, 0).max() * 0.75
    box = (*(centre - half), *(centre + half))
    image = Image.fromarray(np.load(cache / "images.npy", mmap_mode="r")[index]).crop(box).resize((side, side))
    points = (points - (centre - half)) * side / (2 * half)
    draw = ImageDraw.Draw(image)
    for k, ((x, y), colour) in enumerate(zip(points, COLOURS, strict=True), 1):
        draw.ellipse((x - DOT, y - DOT, x + DOT, y + DOT), fill=colour, outline="black")
        draw.text((x + DOT + 2, y - DOT), str(k), fill="white", stroke_width=2, stroke_fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class Labels:
    """The hand labels on disk: {image path: nine [x, y] points in the image's pixels}, saved after every cat."""

    def __init__(self, path: Path):
        self.path = path
        self.points: dict[str, list[list[float]]] = json.loads(path.read_text()) if path.exists() else {}

    def set(self, image: str, points: list[list[float]]) -> None:
        if len(points) != len(NAMES):
            raise ValueError(f"expected {len(NAMES)} points, got {len(points)}")
        self.points[image] = [[round(float(x), 1), round(float(y), 1)] for x, y in points]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.points, indent=1) + "\n")


def handler(afhq: Path, cats: list[str], labels: Labels, guide: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            if self.path == "/":
                self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/state":
                state = {"cats": cats, "labels": labels.points, "names": NAMES, "margin": MARGIN}
                self.reply(json.dumps(state).encode(), "application/json")
            elif self.path == "/reference.png":
                self.reply(guide, "image/png")
            elif self.path.startswith("/image/") and self.path[7:].isdigit() and int(self.path[7:]) < len(cats):
                self.reply((afhq / cats[int(self.path[7:])]).read_bytes(), "image/png")
            else:
                self.reply(b"not found", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/label" or body.get("image") not in cats:
                self.reply(b"bad request", "text/plain", 400)
                return
            labels.set(body["image"], body["points"])
            self.reply(b"{}", "application/json")

        def log_message(self, *_: object) -> None:
            pass

    return Handler


def score(hand: Path, labels: Path) -> dict[str, object]:
    """NME of the model's labels on the hand-labeled cats: the error divided by the hand-placed eyes' distance."""
    truth = json.loads(hand.read_text())
    z = np.load(labels)
    row = {image: i for i, image in enumerate(z["ids"])}
    ids = [image for image in truth if image in row]
    true = np.array([truth[image] for image in ids])
    pred = z["points"][[row[image] for image in ids]]
    eyes = np.linalg.norm(true[:, 0] - true[:, 1], axis=1, keepdims=True)
    err = np.linalg.norm(pred - true, axis=-1) / eyes
    per_face = err.mean(1)
    return {"cats": len(ids), "labels": str(labels), "nme": round(float(per_face.mean()), 4),
            "failure_rate_0.1": round(float((per_face > 0.1).mean()), 4),
            "nme_per_landmark": {n: round(float(v), 4) for n, v in zip(NAMES, err.mean(0), strict=True)}}


def main() -> None:
    """Command line: `serve` the labeling page, or `score` model labels against the hand labels."""
    parser = argparse.ArgumentParser(description="Hand-label AFHQ cats and score the model on them.")
    parser.add_argument("stage", choices=["serve", "score"])
    parser.add_argument("--afhq", type=Path, default=Path("data/afhq"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/cat"))
    parser.add_argument("--out", type=Path, default=Path("data/labels/afhq_hand.json"))
    parser.add_argument("--hand", type=Path, default=Path("data/labels/afhq_hand.json"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels/afhq_cats.npz"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.stage == "score":
        print(json.dumps(score(args.hand, args.labels)), flush=True)
        return
    app = handler(args.afhq, sample(args.afhq), Labels(args.out), reference(args.cache))
    print(f"labeling page at http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), app).serve_forever()


if __name__ == "__main__":
    main()
