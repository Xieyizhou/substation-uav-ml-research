"""Read-only recovery-data assessment; never admits data or inspects protected labels."""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record
from src.vision.collection.simulator_labels import instance_simulator_label

BASE = ROOT / "data/research/ml_training_recovery_v1"
OUTPUT = ROOT / "outputs/research/ml_training_recovery_v1/data-readiness-20260907.json"


def main():
    inputs = {}

    def bind(path):
        inputs[str(path.relative_to(ROOT))] = file_sha256(path)

    def read(path):
        bind(path)
        return read_record(path)

    plans = []
    for path in sorted(BASE.rglob("plan.json")):
        world = path.parent / "world.sdf"
        raw = json.loads(path.read_text())
        if not world.exists() or "annotation_mode" not in raw:
            continue
        plan = read(path)
        bind(world)
        actual = [sensor.findtext("camera/box_type")
                  for sensor in ET.parse(world).iter("sensor")
                  if sensor.get("type") == "boundingbox_camera"]
        plans.append({
            "plan": str(path.relative_to(ROOT)),
            "declared": plan["annotation_mode"], "actual": actual,
            "mode_matches": actual == [plan["annotation_mode"]],
            "world_hash_matches_plan": file_sha256(world) == plan["files"]["world.sdf"],
        })

    capture_checks = []
    for package in ("occlusion-shift-v1", "appearance-background-v1"):
        for path in sorted((BASE / package).glob("*/capture/collection-receipt.json")):
            receipt = read(path)
            plan = read(path.parent.parent / "plan/plan.json")
            rows = []
            for row in receipt["views"]:
                if row["status"] != "captured":
                    continue
                label = instance_simulator_label(row["expected_category"], row["expected_object_id"])
                labels = []
                for obj in row["truth"]["objects"]:
                    match = re.search(r"instance-(\d+)-", obj["annotation_id"])
                    if not match:
                        raise ValueError(f"Missing instance provenance: {row['view_id']}")
                    labels.append(int(match[1]))
                camera = row["camera_position"]
                inside = [obj["name"] for obj in plan["objects"]
                          if all(obj["bounds"][2*i] <= camera[i] <= obj["bounds"][2*i+1]
                                 for i in range(3))]
                rows.append({
                    "view_id": row["view_id"], "expected_object_id": row["expected_object_id"],
                    "expected_class_present": row["expected_class_present"],
                    "planned_instance_present": label in labels,
                    "camera_inside_configured_bounds": inside,
                })
            capture_checks.append({
                "package": package, "variant": path.parent.parent.name,
                "receipt_status": receipt["status"],
                "receipt_matches_plan": receipt["plan_identity"] == plan["identity"],
                "requested": len(plan["calibration_views"]), "captured": len(rows),
                "rejected": sum(row["status"] == "rejected" for row in receipt["views"]),
                "unattempted": len(plan["calibration_views"]) - len(receipt["views"]),
                "planned_instance_absent": sum(not row["planned_instance_present"] for row in rows),
                "camera_inside_bounds": sum(bool(row["camera_inside_configured_bounds"]) for row in rows),
                "rows": rows,
            })

    pools = []
    for name in ("expanded-stratified-v1", "scale-expansion-v1"):
        protocol = read(BASE / name / "protocol.json")
        rows = protocol["rows"]
        pools.append({
            "name": name, "frames": len(rows),
            "unique_image_hashes": len({row["image_sha256"] for row in rows}),
            "negative_frames": sum(not row["truth"]["objects"] for row in rows),
            "by_map": dict(Counter(row["map_id"] for row in rows)),
            "by_source": dict(Counter(row["source"] for row in rows)),
            "class_frame_coverage": dict(Counter(
                category for row in rows
                for category in {obj["class_name"] for obj in row["truth"]["objects"]})),
            "training_admitted": protocol["training_admitted"],
            "promotable": protocol["promotable"],
        })

    variant_manifest = read(BASE / "appearance-background-v1/manifest.json")
    pose_ids = [{row["view_id"] for row in run["selected"]} for run in variant_manifest["runs"]]
    evaluations = {}
    for package in ("targeted-regression-v1", "occlusion-shift-v1", "appearance-background-v1"):
        value = read(BASE / package / "evaluation.json")
        evaluations[package] = {
            "paired_summary": value["paired_summary"],
            "truth_objects": value["summary"]["expanded_uniform_7"]["all"]["truth_objects"],
            "frames": value["rows"],
            "official_v2_11_direct_control_present": any(
                "v2.11" in name for name in value["results"]),
        }
    registry_path = ROOT / "config/perception/real_domain_sources.json"
    bind(registry_path)
    registry = json.loads(registry_path.read_text())
    sources = [{key: source.get(key) for key in (
        "source_id", "ingestion_status", "usage_role", "eligible_partitions", "site_group_count")}
        for source in registry["sources"]]

    for relative in (
        "scripts/vision/audit_recovery_data_readiness.py",
        "scripts/vision/prepare_appearance_background_regression.py",
        "scripts/vision/finalize_appearance_background_review.py",
        "scripts/vision/finalize_occlusion_shifted_review.py",
        "scripts/vision/audit_appearance_background_dedup.py",
        "scripts/vision/evaluate_scale_expansion_recheck.py",
        "scripts/vision/run_recovery_scale_expansion.py",
        "src/vision/canonical/collect.py",
        "src/vision/canonical/occlusion.py",
    ):
        bind(ROOT / relative)
    report = {
        "assessment_date": "2026-09-07", "status": "assessment_complete_with_blockers",
        "plans_checked": len(plans), "mode_mismatches": [p for p in plans if not p["mode_matches"]],
        "plans": plans, "capture_checks": capture_checks, "diagnostic_training_pools": pools,
        "appearance_background_same_pose_overlap": len(pose_ids[0] & pose_ids[1]),
        "reported_evaluations_not_reinferred": evaluations, "real_source_registry": sources,
        "training_admitted": False, "promotable": False, "inputs": inputs,
        "limits": [
            "Recovery subtree only; this is not an inventory of every project dataset.",
            "Read development annotations only; no protected labels, images or predictions accessed.",
            "Configured obstacle bounds are a conservative geometric warning, not a mesh collision proof.",
            "File hashes bind the inspected evidence; image bytes were not rehashed in this assessment.",
            "Absence of a named direct v2.11 control is determined from saved evaluation model keys.",
            "Existing review decisions and frozen historical artifacts are preserved, not re-certified.",
        ],
    }
    report["identity"] = object_sha256(report)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, report)
    print(json.dumps({
        "output": str(OUTPUT), "plans_checked": len(plans),
        "mode_mismatches": report["mode_mismatches"],
        "captures": [{k: v for k, v in row.items() if k != "rows"} for row in capture_checks],
        "pools": pools, "identity": report["identity"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
