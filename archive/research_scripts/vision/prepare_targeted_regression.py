"""Freeze a fresh, development-only targeted regression set."""
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.expansion import target_candidates
from src.vision.canonical.plan import read_record, write_record


BASE = ROOT / "data/research/ml_training_recovery_v1/targeted-regression-v1"
SOURCES = {
    name: ROOT / f"data/research/canonical_views_v1/{name}-occlusion-filtered-plan-v1/plan.json"
    for name in ("medium", "complex")
}
CONFIGS = {
    "medium": ROOT / "config/maps/substation_medium.json",
    "complex": ROOT / "config/maps/substation_complex.json",
}
SPEC = {
    "medium": {"capacitor_bank": 6},
    "complex": {"switchgear": 6, "capacitor_bank": 6, "reactor": 6},
}
ROUND_INDEX = 121


def known_view_ids():
    known = set()
    for receipt_path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(receipt_path):
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        known.update(row.get("view_id") for row in receipt.get("views", []) if row.get("view_id"))
    return known


def choose_targeted(rows, count):
    rows = sorted(rows, key=lambda row: (-row["distance"], row["object_id"], row["bearing"], row["height"], row["offset"], row["view_id"]))
    selected, used_bins = [], set()
    for row in rows:
        key = (row["object_id"], row["bearing"] // 30)
        if key not in used_bins:
            selected.append(row)
            used_bins.add(key)
            if len(selected) == count:
                return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(f"Not enough targeted candidates: need {count}, have {len(rows)}")


def copy_world(source_path, target_dir):
    source = read_record(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in source["files"].items():
        source_file = source_path.parent / name
        if file_sha256(source_file) != digest:
            raise ValueError(f"Changed source artifact: {source_file}")
        (target_dir / name).write_bytes(source_file.read_bytes())
    world = target_dir / "world.sdf"
    tree = ET.parse(world)
    sensors = [sensor for sensor in tree.iter("sensor") if sensor.get("type") == "boundingbox_camera"]
    if len(sensors) != 1:
        raise ValueError(f"Expected one bounding-box camera: {world}")
    box_type = sensors[0].find("camera/box_type")
    if box_type is None:
        raise ValueError(f"Missing camera box_type: {world}")
    box_type.text = "full_2d"
    tree.write(world, encoding="utf-8", xml_declaration=True)
    return source, world


def prepare():
    manifest_path = BASE / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    known = known_view_ids()
    runs = []
    for map_id, source_path in SOURCES.items():
        source = read_record(source_path)
        if source.get("label_mode") != "visual-instance" or source.get("hierarchy_mode") != "top-level-equipment":
            raise ValueError(f"Invalid source label mode: {source_path}")
        config = json.loads(CONFIGS[map_id].read_text())
        candidates = target_candidates(map_id, source["objects"], config["gazebo_world_origin_m"][:2], (config["width"], config["height"]), ROUND_INDEX)
        selected = []
        for category, count in SPEC[map_id].items():
            options = [row for row in candidates if row["category"] == category and row["view_id"] not in known]
            far = [row for row in options if row["distance"] >= 15]
            selected.extend(choose_targeted(far if len(far) >= count else options, count))
        if len({row["view_id"] for row in selected}) != len(selected):
            raise ValueError(f"Duplicate selected view ids: {map_id}")
        plan_dir = BASE / map_id / "plan"
        source_copy, world = copy_world(source_path, plan_dir)
        plan = {key: copy.deepcopy(value) for key, value in source_copy.items() if key != "identity"}
        plan.update(
            annotation_mode="full_2d",
            calibration_views=selected,
            pilot_views=[],
            diagnostic_only=True,
            diagnostic_allow_expected_absence=False,
            diagnostic_require_expected_presence=True,
            training_admitted=False,
            automatic_training=False,
            diagnostic_purpose="fresh_targeted_regression_holdout",
            source_plan_path=str(source_path),
            source_plan_sha256=file_sha256(source_path),
            expansion_round=ROUND_INDEX,
            excluded_known_view_count=len(known),
        )
        plan["files"]["world.sdf"] = file_sha256(world)
        written = write_record(plan_dir / "plan.json", plan)
        runs.append({
            "map_id": map_id,
            "plan_path": str(plan_dir / "plan.json"),
            "plan_identity": written["identity"],
            "selected": [{key: row[key] for key in ("view_id", "category", "object_id", "bearing", "distance", "height", "offset")} for row in selected],
            "candidate_count": len(candidates),
            "unseen_candidate_count": sum(row["view_id"] not in known for row in candidates),
            "training_admitted": False,
            "promotable": False,
        })
    manifest = {
        "status": "prepared",
        "purpose": "fresh_targeted_regression_holdout",
        "round_index": ROUND_INDEX,
        "known_view_count": len(known),
        "runs": runs,
        "selection_rule": "Fresh unseen full_2d target poses matching the prior medium capacitor_bank and complex switchgear/capacitor_bank/reactor distribution; far poses preferred.",
        "training_admitted": False,
        "promotable": False,
    }
    manifest["identity"] = object_sha256(manifest)
    BASE.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    prepare()
