"""Render class-specific review boards from a quarantined YOLO ZIP."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image, ImageDraw, ImageFont
import yaml


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    mapping = json.loads(args.mapping.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as bundle:
        names = _class_names(bundle)
        labels = _label_index(bundle)
        candidates = _candidates(bundle, labels, names, mapping, args.source_id)
        for target, rows in sorted(candidates.items()):
            target_root = args.output / target
            target_root.mkdir(parents=True, exist_ok=True)
            for offset in range(0, len(rows), 64):
                batch = rows[offset:offset + 64]
                number = offset // 64 + 1
                _render_board(bundle, batch, target_root / f"batch-{number:03d}.jpg")
                _write_jsonl(target_root / f"batch-{number:03d}.jsonl", batch)
            _write_jsonl(target_root / "review.jsonl", rows)


def _class_names(bundle):
    data = yaml.safe_load(bundle.read("data.yaml"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] if index in names else names[str(index)] for index in range(len(names))]
    return names


def _label_index(bundle):
    return {(_split(path), path.stem): name for name in bundle.namelist()
            if (path := PurePosixPath(name)).suffix.lower() == ".txt"
            and "labels" in path.parts and _split(path) is not None}


def _split(path):
    return next((part for part in path.parts if part in {"train", "valid", "test"}), None)


def _candidates(bundle, labels, names, mapping, source_id):
    result, seen = defaultdict(list), defaultdict(set)
    for image_name in bundle.namelist():
        path = PurePosixPath(image_name)
        split = _split(path)
        if split is None or "images" not in path.parts or path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
        }:
            continue
        label_name = labels.get((split, path.stem))
        if label_name is None:
            continue
        payload = bundle.read(image_name)
        digest = sha256(payload).hexdigest()
        rows = bundle.read(label_name).decode().splitlines()
        boxes = defaultdict(list)
        for line in rows:
            values = line.split()
            if len(values) < 5 or len(values) % 2 == 0:
                continue
            label = names[int(float(values[0]))]
            target = mapping.get(label)
            if target:
                coordinates = [float(value) for value in values[1:]]
                if len(coordinates) == 4:
                    box = coordinates
                else:
                    xs, ys = coordinates[::2], coordinates[1::2]
                    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                    box = [(left + right) / 2, (top + bottom) / 2,
                           right - left, bottom - top]
                boxes[target].append(box)
        for target, target_boxes in boxes.items():
            if digest in seen[target]:
                continue
            seen[target].add(digest)
            result[target].append({
                "candidate_id": f"{source_id}-{target}-{digest[:16]}",
                "source_id": source_id, "target_class": target,
                "source_path": image_name, "source_split_untrusted": split,
                "image_sha256": digest, "boxes": target_boxes,
                "review_status": "pending", "semantic_decision": "ambiguous",
                "whole_equipment_confirmed": None, "review_note": "",
            })
    return result


def _render_board(bundle, rows, output):
    cards = []
    for index, row in enumerate(rows, start=1):
        image = Image.open(BytesIO(bundle.read(row["source_path"]))).convert("RGB")
        image.thumbnail((210, 130))
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for x, y, w, h in row["boxes"]:
            draw.rectangle(((x-w/2)*width, (y-h/2)*height,
                            (x+w/2)*width, (y+h/2)*height), outline="#ff3b21", width=3)
        card = Image.new("RGB", (220, 170), "#151a18")
        card.paste(image, ((220-width)//2, 4))
        caption = f"{index:02d} {row['candidate_id'][-8:]}\n{PurePosixPath(row['source_path']).name[:28]}"
        ImageDraw.Draw(card).multiline_text((5, 138), caption, fill="#f4f0e6",
                                            font=ImageFont.load_default(), spacing=2)
        cards.append(card)
    sheet = Image.new("RGB", (1760, 1360), "#e9e3d6")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 8) * 220, (index // 8) * 170))
    sheet.save(output, quality=92)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
