"""Render and explicitly finalize the 66-frame common trusted training base."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json

SOURCE = ROOT / "data/research/ml_training_recovery_v1/expanded-stratified-v1"
OUT = ROOT / "data/research/ml_training_recovery_v1/trusted-training-base-v1"
COLORS = {"transformer":"#ff3b30", "switchgear":"#00a8ff", "capacitor_bank":"#34c759", "reactor":"#ffcc00"}


def load_rows():
    protocol_path = SOURCE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    rows = protocol["rows"]
    if len(rows) != 66 or len({row["view_id"] for row in rows}) != 66:
        raise ValueError("Expected exactly 66 unique common-base frames")
    return protocol, rows, protocol_path


def target_box(row):
    candidates = [obj for obj in row["truth"]["objects"] if obj["class_name"] == row["expected_category"]]
    if not candidates:
        raise ValueError(f"Planned category missing from truth: {row['view_id']}")
    # The source rows do not preserve the instance-label mapping; the sampling anchor
    # is used only to select the largest same-class box for review evidence.
    return max(candidates, key=lambda obj: (obj["bbox_xyxy"][2]-obj["bbox_xyxy"][0])*(obj["bbox_xyxy"][3]-obj["bbox_xyxy"][1]))["bbox_xyxy"]


def tile(row):
    image = Image.open(row["rgb_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for obj in row["truth"]["objects"]:
        draw.rectangle(tuple(obj["bbox_xyxy"]), outline=COLORS[obj["class_name"]], width=4)
    box = target_box(row)
    draw.rectangle(tuple(box), outline="white", width=7)
    full = image.copy(); full.thumbnail((480, 260))
    x0,y0,x1,y1 = box
    crop = Image.open(row["rgb_path"]).convert("RGB").crop((max(0,x0-30),max(0,y0-30),min(1920,x1+30),min(1080,y1+30)))
    crop.thumbnail((220, 90))
    cell = Image.new("RGB", (480, 360), "#202124")
    cell.paste(full, ((480-full.width)//2, 24))
    cell.paste(crop, ((480-crop.width)//2, 266))
    ImageDraw.Draw(cell).text((5,5), f"{row['source']} | {row['map_id']} | {row['expected_category']} | {row['view_id'][:10]}", fill="white")
    return cell, box


def render():
    protocol, rows, protocol_path = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    frames, pages = [], []
    for page_index in range(6):
        members = rows[page_index*12:(page_index+1)*12]
        page = Image.new("RGB", (1920, 1080), "#111")
        for index, row in enumerate(members):
            cell, box = tile(row)
            page.paste(cell, ((index%4)*480, (index//4)*360))
            training_image = Path(row["training_path"])
            label = SOURCE / "labels" / f"{training_image.stem}.txt"
            if file_sha256(Path(row["rgb_path"])) != row["image_sha256"]:
                raise ValueError("Source RGB changed before review")
            frames.append({
                "view_id": row["view_id"], "source": row["source"], "map_id": row["map_id"],
                "expected_category": row["expected_category"], "expected_object_id": row["expected_object_id"],
                "source_image_path": row["rgb_path"], "source_image_sha256": row["image_sha256"],
                "training_image_path": str(training_image), "training_image_sha256": file_sha256(training_image),
                "training_label_path": str(label), "training_label_sha256": file_sha256(label),
                "truth_sha256": object_sha256(row["truth"]), "target_review_box_xyxy": box,
            })
        path = OUT / f"review-page-{page_index+1}.png"
        page.save(path)
        pages.append({"path":str(path), "sha256":file_sha256(path), "view_ids":[row["view_id"] for row in members]})
    manifest = {
        "status":"rendered_pending_ai_review", "source_protocol_identity":protocol["identity"],
        "source_protocol_sha256":file_sha256(protocol_path), "frames":frames, "pages":pages,
        "training_admitted":False, "promotable":False,
    }
    manifest["identity"] = object_sha256(manifest)
    write_json(OUT / "review-manifest.json", manifest)
    return manifest


def finalize(manifest, accepted_ids):
    expected = {row["view_id"] for row in manifest["frames"]}
    if expected != set(accepted_ids) or len(expected) != 66:
        raise ValueError("Explicit common-base review decisions are incomplete")
    entries = []
    for row in manifest["frames"]:
        for path_key, hash_key in (("source_image_path","source_image_sha256"),("training_image_path","training_image_sha256"),("training_label_path","training_label_sha256")):
            if file_sha256(Path(row[path_key])) != row[hash_key]:
                raise ValueError(f"Reviewed input changed: {row[path_key]}")
        entries.append({**row, "data_role":"trusted_training_base", "review_decision":"accepted",
            "review_nature":"AI-assisted", "reviewer":"codex_visual_inspection_2026-09-07",
            "review_reason":"Full-frame labels and planned-target crop inspected; image is usable and visible taxonomy objects are consistently boxed.",
            "reviewed_at":"2026-09-07T00:00:00+08:00", "training_admitted":False, "promotable":False})
    ledger = {
        "schema_version":1, "dataset_version":"trusted-training-base-v1", "status":"frozen",
        "frame_count":66, "source_protocol_identity":manifest["source_protocol_identity"],
        "source_review_manifest_identity":manifest["identity"], "entries":entries,
        "training_admitted":False, "promotable":False,
    }
    ledger["identity"] = object_sha256(ledger)
    write_json(OUT / "frozen-ledger.json", ledger)
    return ledger


if __name__ == "__main__":
    result = render()
    print(json.dumps({"identity":result["identity"], "pages":result["pages"]}, indent=2))
