"""Freeze a 24-frame canonical-only hard-negative replacement matrix."""
import copy
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.expansion import _pose
from src.vision.canonical.gates import annotation_mode_from_world, validate_view_pose
from src.vision.canonical.occlusion import view_blockers
from src.vision.canonical.plan import read_record, write_record
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE as SUPPLEMENT

BASE = SUPPLEMENT / "negative-redesign-v2"
PAUSED = ROOT / "data/research/background_calibration_v1/PAUSED.json"
ROUND = 281
TARGETS = {"transformer", "switchgear", "capacitor_bank", "reactor"}
LIGHTS = ("light_normal", "light_cool_low")
SOURCES = {
    "simple": ROOT / "data/research/canonical_views_v1/simple-occlusion-filtered-plan-v1/plan.json",
    "medium": ROOT / "data/research/canonical_views_v1/medium-occlusion-filtered-plan-v1/plan.json",
    "complex": ROOT / "data/research/canonical_views_v1/complex-occlusion-filtered-plan-v1/plan.json",
}
CONFIGS = {
    "simple": ROOT / "config/maps/substation_training.json",
    "medium": ROOT / "config/maps/substation_medium.json",
    "complex": ROOT / "config/maps/substation_complex.json",
}
FAMILIES = {
    "simple": (("simple_cabinet_array", "cabinet_2"), ("simple_utility_pole", "pole_east_2")),
    "medium": (("medium_ordinary_cabinet", "cabinet_west"), ("medium_utility_pole", "pole_c")),
    "complex": (("complex_control_building", "control_building"), ("complex_ordinary_cabinet", "cabinet_center")),
}


def apply_light(world, light_id):
    if light_id == "light_normal":
        return
    if light_id != "light_cool_low":
        raise ValueError(f"Unknown lighting variant: {light_id}")
    scene = world.find("scene")
    if scene is None:
        scene = ET.SubElement(world, "scene")
    ambient = scene.find("ambient")
    if ambient is None:
        ambient = ET.SubElement(scene, "ambient")
    sun_diffuse = world.find("light[@name='sun']/diffuse")
    if sun_diffuse is None:
        raise ValueError("Canonical world is missing the sun diffuse field")
    ambient.text = "0.30 0.34 0.40 1"
    sun_diffuse.text = "0.48 0.56 0.68 1"


def select_poses(map_id, source, config, family, object_id):
    obj = next(row for row in source["objects"] if row["name"] == object_id)
    bounds = obj["bounds"]
    target = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    candidates = []
    for step in range(48):
        bearing = (ROUND + step * 37 + int(object_sha256(family)[:4], 16)) % 360
        distance = (5.25, 6.75, 8.25, 9.75)[step % 4]
        height = (2.1, 2.8, 3.5)[step % 3]
        angle = math.radians(bearing)
        camera = [target[0] + distance * math.cos(angle), target[1] + distance * math.sin(angle), height]
        position, orientation = _pose(camera, target, 0)
        row = {
            "map_id": map_id,
            "object_id": object_id,
            "category": "no_target",
            "hard_negative_type": obj["category"],
            "subject_family": family,
            "bearing": bearing,
            "distance": distance,
            "height": height,
            "offset": 0,
            "camera_position": camera,
            "position": position,
            "orientation": orientation,
            "family": f"{map_id}:negative-bridge:{family}:{bearing}",
        }
        row["pose_id"] = object_sha256({"negative_bridge_round": ROUND, **row})
        try:
            validate_view_pose(row, config)
        except ValueError:
            continue
        if not view_blockers(row, source["objects"]):
            candidates.append(row)
    candidates.sort(key=lambda row: (-row["distance"], row["bearing"], row["pose_id"]))
    if not candidates:
        raise ValueError(f"No legal pose for {family}")
    selected = [candidates[0]]
    reference_bearing = selected[0]["bearing"]
    selected.extend(
        row for row in candidates[1:]
        if min(abs(row["bearing"] - reference_bearing) % 360,
               360 - abs(row["bearing"] - reference_bearing) % 360) >= 75
    )
    if len(selected) < 2:
        raise ValueError(f"No directionally separated second pose for {family}")
    return [copy.deepcopy(row) for row in selected[:2]]


def make_plan(map_id, source_path, source, base_views, light_id):
    run_id = f"{map_id}-{light_id}"
    folder = BASE / "runs" / run_id / "plan"
    folder.mkdir(parents=True, exist_ok=False)
    for name, digest in source["files"].items():
        src = source_path.parent / name
        if file_sha256(src) != digest:
            raise ValueError(f"Canonical source artifact changed: {src}")
        shutil.copyfile(src, folder / name)
    world_path = folder / "world.sdf"
    tree = ET.parse(world_path)
    sensors = [row for row in tree.iter("sensor") if row.get("type") == "boundingbox_camera"]
    if len(sensors) != 1 or sensors[0].find("camera/box_type") is None:
        raise ValueError("Expected exactly one declared bounding-box camera")
    sensors[0].find("camera/box_type").text = "full_2d"
    world = tree.getroot().find("world")
    target_names = {row["name"] for row in source["objects"] if row["category"] in TARGETS}
    removed = []
    for model in list(world.findall("model")):
        if model.get("name") in target_names:
            removed.append(model.get("name"))
            world.remove(model)
    if set(removed) != target_names:
        raise ValueError(f"Target isolation mismatch for {map_id}")
    apply_light(world, light_id)
    ET.indent(tree, space="  ")
    tree.write(world_path, encoding="utf-8", xml_declaration=True)
    if annotation_mode_from_world(world_path) != "full_2d":
        raise ValueError("Persisted world did not retain full_2d")

    views = []
    for base in base_views:
        row = copy.deepcopy(base)
        row.update(
            view_id=object_sha256({"pose_id": base["pose_id"], "light_id": light_id, "revision": 2}),
            pair_id=object_sha256({"pose_id": base["pose_id"], "negative_bridge_revision": 2}),
            derivation_group=f"negative-bridge-v2:{base['pose_id']}",
            lighting_id=light_id,
            material_id="material_original",
            data_role="new_training_candidate",
            training_admitted=False,
            promotable=False,
        )
        views.append(row)
    record = {key: copy.deepcopy(value) for key, value in source.items() if key != "identity"}
    record.update(
        annotation_mode="full_2d",
        label_mode="visual-instance",
        hierarchy_mode="top-level-equipment",
        calibration_views=views,
        pilot_views=[],
        diagnostic_only=True,
        diagnostic_allow_expected_absence=True,
        diagnostic_require_expected_presence=False,
        diagnostic_require_no_targets=True,
        diagnostic_purpose="visual_bridge_canonical_hard_negative_v2",
        source_layout_id=f"{map_id}-canonical-v1",
        derived_layout_id=f"{map_id}-canonical-target-isolated-negative-v2",
        removed_target_models=sorted(removed),
        lighting_id=light_id,
        material_id="material_original",
        recording_group=run_id,
        training_admitted=False,
        promotable=False,
        automatic_training=False,
    )
    record["files"]["world.sdf"] = file_sha256(world_path)
    return write_record(folder / "plan.json", record), folder


def prepare():
    path = BASE / "matrix.json"
    if path.exists():
        return json.loads(path.read_text())
    paused = json.loads(PAUSED.read_text())
    if paused.get("collection_allowed") is not False:
        raise ValueError("Expected the off-scope background package to remain collection-blocked")

    runs = []
    entries = []
    inputs = {str(PAUSED): file_sha256(PAUSED), str(Path(__file__)): file_sha256(Path(__file__))}
    for map_id, source_path in SOURCES.items():
        source = read_record(source_path)
        config = json.loads(CONFIGS[map_id].read_text())
        inputs[str(source_path)] = file_sha256(source_path)
        inputs[str(CONFIGS[map_id])] = file_sha256(CONFIGS[map_id])
        base_views = []
        for family, object_id in FAMILIES[map_id]:
            base_views.extend(select_poses(map_id, source, config, family, object_id))
        for light_id in LIGHTS:
            plan, folder = make_plan(map_id, source_path, source, base_views, light_id)
            runs.append({
                "run_id": f"{map_id}-{light_id}",
                "map_id": map_id,
                "lighting_id": light_id,
                "frame_count": 4,
                "plan_path": str(folder / "plan.json"),
                "plan_identity": plan["identity"],
                "world_sha256": plan["files"]["world.sdf"],
                "actual_annotation_mode": annotation_mode_from_world(folder / "world.sdf"),
            })
            for row in plan["calibration_views"]:
                entries.append({
                    "candidate_id": object_sha256({"run": f"{map_id}-{light_id}", "view": row["view_id"]}),
                    "view_id": row["view_id"],
                    "pair_id": row["pair_id"],
                    "derivation_group": row["derivation_group"],
                    "map_layout_id": plan["derived_layout_id"],
                    "source_layout_id": plan["source_layout_id"],
                    "subject_family": row["subject_family"],
                    "subject_instance": row["object_id"],
                    "pose_id": row["pose_id"],
                    "lighting_id": light_id,
                    "capture_status": "planned",
                    "review_decision": "pending",
                    "training_admitted": False,
                    "promotable": False,
                })
    pair_counts = Counter(row["pair_id"] for row in entries)
    if len(entries) != 24 or len(pair_counts) != 12 or set(pair_counts.values()) != {2}:
        raise ValueError("Negative replacement is not 12 paired poses / 24 frames")
    matrix = {
        "schema_version": 1,
        "status": "frozen_pending_pilot",
        "dataset_version": "visual-bridge-negative-redesign-v2",
        "supersedes_matrix_identity": json.loads((SUPPLEMENT / "matrix.json").read_text())["identity"],
        "supersedes_negative_scope_only": True,
        "supersession_reason": "The earlier negative design referenced a package later marked collection_allowed=false because it is outside canonical product-world scope.",
        "counts": {"frames": 24, "independent_pose_groups": 12, "lights_per_pose": 2},
        "family_counts": dict(sorted(Counter(row["subject_family"] for row in entries).items())),
        "runs": runs,
        "entries": entries,
        "pilot_rule": "One pose from each family under both lights: 12 frames before continuation.",
        "allowed_world_changes": ["remove configured four-class target models", "bounding-box camera box_type to full_2d", "ambient and sun diffuse for light_cool_low"],
        "off_scope_background_package": {"status": paused["status"], "collection_allowed": paused["collection_allowed"], "identity": paused["identity"]},
        "inputs": inputs,
        "unseen_scene_status": "sealed_not_evaluated",
        "training_admitted": False,
        "promotable": False,
    }
    matrix["identity"] = object_sha256(matrix)
    write_json(path, matrix)
    return matrix


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({k: result[k] for k in ("status", "counts", "family_counts", "off_scope_background_package", "identity")}, indent=2))
