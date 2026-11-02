"""Generate tiny geometric format fixtures; these are not equipment photographs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

CLASSES = ("transformer", "switchgear", "capacitor_bank", "reactor")
WIDTH, HEIGHT = 320, 240


def encoded_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def png(index, boxes):
    colors = ((48, 133, 120), (58, 99, 160), (210, 149, 48), (151, 80, 135))
    pixels = bytearray()
    for y in range(HEIGHT):
        pixels.append(0)
        for x in range(WIDTH):
            shade = 242 + (index + x // 40 + y // 40) % 10
            color = (shade, shade, shade)
            for class_id, (left, top, right, bottom) in enumerate(boxes):
                if left <= x < right and top <= y < bottom:
                    color = colors[class_id]
            pixels.extend(color)

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(pixels), level=9))
            + chunk(b"IEND", b""))


def fixture_files():
    files = {}
    frames = []
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    files["dataset/dataset.yaml"] = (
        f"path: .\ntrain: images/train\nval: images/validation\nnames:\n{names}\n"
    ).encode()
    for index in range(12):
        dx, dy = index % 4 * 3, index // 4 * 2
        boxes = [(30+dx, 30+dy, 90+dx, 80+dy),
                 (190-dx, 35+dy, 218-dx, 59+dy),
                 (35+dx, 145-dy, 110+dx, 190-dy),
                 (215-dx, 155-dy, 240-dx, 180-dy)]
        split = "train" if index < 8 else "validation"
        stem = f"format-{index:02d}"
        image_path = f"dataset/images/{split}/{stem}.png"
        files[image_path] = png(index, boxes)
        lines, truth = [], []
        for class_id, box in enumerate(boxes):
            left, top, right, bottom = box
            values = ((left+right)/2/WIDTH, (top+bottom)/2/HEIGHT,
                      (right-left)/WIDTH, (bottom-top)/HEIGHT)
            lines.append(str(class_id) + " " + " ".join(f"{value:.8f}" for value in values))
            truth.append(dict(class_name=CLASSES[class_id], bbox=box,
                              small=(right-left)*(bottom-top) < 32**2))
        files[f"dataset/labels/{split}/{stem}.txt"] = ("\n".join(lines) + "\n").encode()
        if split == "validation":
            frames.append(dict(
                sample_id=stem, image_relative_path=image_path, width=WIDTH, height=HEIGHT,
                image_sha256=hashlib.sha256(files[image_path]).hexdigest(), truth=truth,
            ))
    for name in ("baseline", "candidate"):
        rows = []
        for index, frame in enumerate(frames):
            predictions = []
            for class_id, item in enumerate(frame["truth"]):
                omitted = ((name == "baseline" and class_id == 3)
                           or (name == "baseline" and class_id == 1 and index % 2 == 0)
                           or (name == "candidate" and class_id == 1 and index == 0))
                if not omitted:
                    predictions.append(dict(class_name=item["class_name"], bbox=item["bbox"], confidence=0.9))
            if name == "baseline" or index == 3:
                predictions.append(dict(class_name="reactor", bbox=[135, 110, 158, 132], confidence=0.8))
            rows.append(dict(frame, predictions=predictions))
        files[name + ".json"] = encoded_json(dict(
            prediction_fixture_schema_version=1, dataset_role="demo", formal_evidence=False,
            prediction_origin="manually_constructed_cached_predictions",
            description="Geometric format fixtures. No model inference or real equipment accuracy is represented.",
            frames=rows,
        ))
    files["PROVENANCE.json"] = encoded_json(dict(
        fixture_schema_version=1, dataset_role="demo", formal_evidence=False,
        image_origin="project_generated_geometric_rectangles",
        prediction_origin="manually_constructed_cached_predictions",
        generator="examples/generate_fixtures.py", license="MIT",
        equipment_photographs=False, independently_reviewed_labels=False,
        image_count=12, split_counts=dict(train=8, validation=4), class_names=list(CLASSES),
        files=[dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
               for name, payload in sorted(files.items())],
    ))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "fixtures")
    parser.add_argument("--check", action="store_true", help="Verify fixtures without writing them.")
    args = parser.parse_args()
    files = fixture_files()
    for name, payload in files.items():
        destination = args.output / name
        if args.check:
            if not destination.is_file() or destination.read_bytes() != payload:
                raise ValueError("Fixture differs: " + name)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != payload:
                raise FileExistsError("Refusing to replace a different fixture: " + str(destination))
            destination.write_bytes(payload)
    print(json.dumps(dict(files=len(files), checked=args.check, dataset_role="demo", formal_evidence=False)))


if __name__ == "__main__":
    main()
