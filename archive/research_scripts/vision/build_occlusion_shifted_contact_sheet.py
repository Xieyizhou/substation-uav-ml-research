"""Build a labeled contact sheet for occlusion-shifted semantic review."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/research/ml_training_recovery_v1/occlusion-shift-v1"


def main():
    rows = []
    for receipt_path in sorted(BASE.glob("*/capture/collection-receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        plan = json.loads((receipt_path.parent.parent / "plan/plan.json").read_text())
        selected = {row["view_id"]: row for row in plan["calibration_views"]}
        for row in receipt["views"]:
            if row.get("status") == "captured":
                rows.append({**row, "occlusion_blockers": selected[row["view_id"]].get("occlusion_blockers", [])})
    rows.sort(key=lambda row: (row["map_id"], row["expected_category"], row["view_id"]))
    if len(rows) != 24:
        raise ValueError(f"Expected 24 captured rows, got {len(rows)}")
    cols, tile_w, tile_h, image_h = 6, 320, 280, 225
    sheet = Image.new("RGB", (cols * tile_w, ((len(rows) + cols - 1) // cols) * tile_h), "white")
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    for index, row in enumerate(rows):
        image = Image.open(row["rgb_path"]).convert("RGB")
        image.thumbnail((tile_w - 8, image_h - 8))
        x, y = (index % cols) * tile_w, (index // cols) * tile_h
        sheet.paste(image, (x + (tile_w - image.width) // 2, y + 4))
        blockers = ",".join(row["occlusion_blockers"])
        draw.text((x + 4, y + image_h + 7), f"{row['map_id']} | {row['expected_category']} | {row['view_id'][:8]}", fill="black", font=font)
        draw.text((x + 4, y + image_h + 21), f"blockers={blockers[:42]}", fill="black", font=font)
        draw.text((x + 4, y + image_h + 35), f"present={row.get('expected_class_present')}", fill="black", font=font)
    output = BASE / "contact-sheet.png"; sheet.save(output)
    print(json.dumps({"output": str(output), "frames": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
