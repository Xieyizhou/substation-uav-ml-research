"""Prepare a development-only full_2d expansion across canonical maps.

The existing 30-image diagnostic pool is too small for the missing map/class
strata. This script derives new camera poses from the already materialized
canonical worlds, keeps visual-instance/top-level-equipment labels, and
freezes a balanced candidate set before capture. It never admits data to
formal training.
"""
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


BASE = ROOT / "data/research/ml_training_recovery_v1/stratified-expansion-v1"
SOURCES = {
    name: ROOT / f"data/research/canonical_views_v1/{name}-occlusion-filtered-plan-v1/plan.json"
    for name in ("simple", "medium", "complex")
}
MAP_CONFIGS = {
    "simple": ROOT / "config/substation_obstacles.json",
    "medium": ROOT / "config/maps/substation_medium.json",
    "complex": ROOT / "config/maps/substation_complex.json",
}
TARGETS = ("transformer", "switchgear", "capacitor_bank", "reactor")
PER_CLASS = {"simple": 4, "medium": 4, "complex": 4}
ROUND_INDEX = 41


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


def choose_spread(rows, count):
    """Choose deterministic poses spread across object, bearing, distance, height."""
    rows = sorted(rows, key=lambda row: (row["object_id"], row["bearing"], row["distance"], row["height"], row["offset"], row["view_id"]))
    if len(rows) < count:
        raise ValueError(f"Not enough unseen candidates: need {count}, have {len(rows)}")
    step = len(rows) / count
    selected = [rows[min(len(rows) - 1, int(i * step))] for i in range(count)]
    if len({row["view_id"] for row in selected}) != count:
        raise ValueError("Candidate spread selected duplicate view ids")
    return selected


def copy_and_fix_world(source_path, target_dir):
    source = read_record(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in source["files"].items():
        source_file = source_path.parent / name
        if file_sha256(source_file) != digest:
            raise ValueError(f"Source artifact changed: {source_file}")
        (target_dir / name).write_bytes(source_file.read_bytes())
    world = target_dir / "world.sdf"
    tree = ET.parse(world)
    sensors = [s for s in tree.iter("sensor") if s.get("type") == "boundingbox_camera"]
    if len(sensors) != 1:
        raise ValueError(f"Expected one bounding-box camera in {world}")
    box_type = sensors[0].find("camera/box_type")
    if box_type is None:
        raise ValueError(f"Missing camera box_type in {world}")
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
            raise ValueError(f"Source plan is not visual-instance/top-level-equipment: {source_path}")
        config = json.loads(MAP_CONFIGS[map_id].read_text())
        origin = config["gazebo_world_origin_m"][:2]
        size = (config["width"], config["height"])
        candidates = target_candidates(map_id, source["objects"], origin, size, ROUND_INDEX)
        unseen = [row for row in candidates if row["view_id"] not in known]
        selected = []
        for category in TARGETS:
            options = [row for row in unseen if row["category"] == category]
            if options:
                selected.extend(choose_spread(options, PER_CLASS[map_id]))
        if not selected:
            raise ValueError(f"No unseen target candidates for {map_id}")

        map_dir = BASE / map_id
        plan_dir = map_dir / "plan"
        source_copy, world = copy_and_fix_world(source_path, plan_dir)
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
            diagnostic_purpose="stratified_coverage_expansion",
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
            "selected": [
                {
                    "view_id": row["view_id"],
                    "category": row["category"],
                    "object_id": row["object_id"],
                    "bearing": row["bearing"],
                    "distance": row["distance"],
                    "height": row["height"],
                    "offset": row["offset"],
                }
                for row in selected
            ],
            "candidate_count": len(candidates),
            "unseen_candidate_count": len(unseen),
            "training_admitted": False,
            "promotable": False,
        })

    manifest = {
        "status": "prepared",
        "purpose": "development_only_stratified_coverage_expansion",
        "round_index": ROUND_INDEX,
        "known_view_count": len(known),
        "runs": runs,
        "selection_rule": "Four deterministic unseen target poses per available class and map; no background or protected data.",
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
