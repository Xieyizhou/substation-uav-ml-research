"""Prepare a corrected simple-scene full_2d diagnostic plan.

The legacy simple plan used source-level labels. This plan reuses the already
materialized visual-instance/top-level-equipment scene and selects new pilot
views, while keeping the result development-only.
"""
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record, write_record

BASE = ROOT / "data/research/ml_training_recovery_v1/simple-annotation-repair-v2"
SOURCE = ROOT / "data/research/canonical_views_v1/simple-occlusion-filtered-plan-v1/plan.json"


def known_view_ids():
    known = set()
    for receipt_path in (ROOT / "data/research/ml_training_recovery_v1").rglob("collection-receipt.json"):
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        known.update(v.get("view_id") for v in receipt.get("views", []))
    return known


def choose_views(source, known):
    pilot = list(source["pilot_views"])
    switches = [v for v in pilot if v["category"] == "switchgear" and v["view_id"] not in known]
    transformers = [v for v in pilot if v["category"] == "transformer" and v["view_id"] not in known]
    if len(switches) < 5 or len(transformers) < 7:
        raise ValueError("Corrected simple plan does not have enough unseen target pilot views")
    return switches[:5] + transformers[:7]


def prepare():
    existing_manifest = BASE / "manifest.json"
    existing_plan = BASE / "plan/plan.json"
    if existing_manifest.exists() and existing_plan.exists():
        manifest = json.loads(existing_manifest.read_text())
        plan = read_record(existing_plan)
        if manifest.get("plan_identity") != plan.get("identity"):
            raise ValueError("Existing repair manifest does not match its plan")
        return manifest
    source = read_record(SOURCE)
    if source.get("label_mode") != "visual-instance" or source.get("hierarchy_mode") != "top-level-equipment":
        raise ValueError("Source plan is not the corrected visual-instance/top-level-equipment plan")
    known = known_view_ids()
    selected = choose_views(source, known)
    plan_dir = BASE / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in source["files"].items():
        src = SOURCE.parent / name
        if file_sha256(src) != digest:
            raise ValueError(f"Source artifact changed: {name}")
        (plan_dir / name).write_bytes(src.read_bytes())
    world = plan_dir / "world.sdf"
    tree = ET.parse(world)
    sensors = [s for s in tree.iter("sensor") if s.get("type") == "boundingbox_camera"]
    if len(sensors) != 1:
        raise ValueError("Expected exactly one bounding-box camera")
    sensors[0].find("camera/box_type").text = "full_2d"
    tree.write(world, encoding="utf-8", xml_declaration=True)
    plan = {k: copy.deepcopy(v) for k, v in source.items() if k != "identity"}
    plan.update(
        annotation_mode="full_2d",
        calibration_views=selected,
        pilot_views=[],
        diagnostic_only=True,
        diagnostic_allow_expected_absence=False,
        diagnostic_require_expected_presence=True,
        training_admitted=False,
        automatic_training=False,
        diagnostic_purpose="simple_annotation_mode_repair",
        source_plan_path=str(SOURCE),
        source_plan_sha256=file_sha256(SOURCE),
        selected_view_ids=[v["view_id"] for v in selected],
        excluded_known_view_count=len(known),
    )
    plan["files"]["world.sdf"] = file_sha256(world)
    written = write_record(plan_dir / "plan.json", plan)
    manifest = {
        "status": "prepared",
        "plan_path": str(plan_dir / "plan.json"),
        "plan_identity": written["identity"],
        "source_plan": str(SOURCE),
        "selected": [{"view_id": v["view_id"], "category": v["category"], "object_id": v["object_id"], "bearing": v.get("bearing"), "distance": v.get("distance"), "height": v.get("height")} for v in selected],
        "training_admitted": False,
        "promotable": False,
    }
    manifest["identity"] = object_sha256(manifest)
    write_json(BASE / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    prepare()
