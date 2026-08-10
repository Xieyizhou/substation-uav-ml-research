"""Ordered batch-one image sources for static ONNX replay."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def ordered_replay_sources(rows, heldout_root):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = []
        for index, row in enumerate(rows):
            source = (Path(heldout_root) / row["image_relative_path"]).resolve()
            destination = root / f"{index:08d}{source.suffix.lower()}"
            destination.symlink_to(source)
            paths.append(destination)
        yield root, paths


def write_source_list(root, name, paths):
    path = Path(root) / f"{name}.txt"
    path.write_text(
        "".join(f"{Path(item).absolute()}\n" for item in paths),
        encoding="utf-8",
    )
    return path


def static_predict_options(condition):
    return {
        "stream": True, "batch": 1, "imgsz": condition.input_width,
        "conf": condition.confidence_threshold, "iou": 0.7,
        "device": "cpu", "rect": False, "verbose": False,
    }
