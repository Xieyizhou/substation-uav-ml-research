"""Render a stratified Gomes transformer review contact sheet."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont


def main(argv=None):
    manifest, yolo_root, classes_path, output = map(Path, (argv or sys.argv[1:]))
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    transformer_id = classes.index("Power transformer")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["members"][0]["archive"]].append(row)
    selected = [row for archive in sorted(grouped) for row in grouped[archive][:6]]
    cards = [_render_card(row, yolo_root, transformer_id) for row in selected]
    sheet = Image.new("RGB", (4 * 420, 6 * 300), "#eee9dd")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 4) * 420, (index // 4) * 300))
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / "batch-001.jpg", quality=92)
    with (output / "batch-001.jsonl").open("w", encoding="utf-8") as destination:
        for index, row in enumerate(selected, start=1):
            record = dict(row)
            record["batch_item"] = index
            destination.write(json.dumps(record, sort_keys=True) + "\n")


def _render_card(row, root, transformer_id):
    member = row["members"][0]
    with zipfile.ZipFile(root / member["archive"]) as bundle:
        image = Image.open(BytesIO(bundle.read(member["member"]))).convert("RGB")
        stem = PurePosixPath(member["member"]).stem
        label_name = next(name for name in bundle.namelist()
                          if PurePosixPath(name).stem == stem
                          and PurePosixPath(name).suffix.lower() == ".txt"
                          and "labels" in PurePosixPath(name).parts)
        labels = bundle.read(label_name).decode().splitlines()
    image.thumbnail((400, 245))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for line in labels:
        values = line.split()
        if int(float(values[0])) != transformer_id:
            continue
        x, y, w, h = map(float, values[1:])
        box = ((x - w / 2) * width, (y - h / 2) * height,
               (x + w / 2) * width, (y + h / 2) * height)
        draw.rectangle(box, outline="#ff3b21", width=4)
    card = Image.new("RGB", (420, 300), "#171b19")
    card.paste(image, ((420 - width) // 2, 8))
    caption = f"{row['candidate_id']}\n{member['archive']} | {PurePosixPath(member['member']).name}"
    ImageDraw.Draw(card).multiline_text((10, 258), caption, fill="#f4f0e6",
                                        font=ImageFont.load_default(), spacing=3)
    return card


if __name__ == "__main__":
    main()
