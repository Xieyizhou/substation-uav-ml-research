"""Fail-closed semantic gates for canonical offline camera collection."""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from src.ml.artifacts import file_sha256

CHECK_VERSION = "canonical-collection-gates-v2"
REQUIRED_ANNOTATION_MODE = "full_2d"
REQUIRED_LABEL_MODE = "visual-instance"
REQUIRED_HIERARCHY_MODE = "top-level-equipment"
OBSTACLE_CLEARANCE_M = 0.25


def annotation_mode_from_world(path):
    sensors = [s for s in ET.parse(path).iter("sensor")
               if s.get("type") == "boundingbox_camera"]
    if len(sensors) != 1:
        raise ValueError(f"Expected exactly one bounding-box camera, found {len(sensors)}")
    mode = sensors[0].findtext("camera/box_type")
    if not mode:
        raise ValueError("Bounding-box camera is missing box_type")
    if mode not in {"full_2d", "visible_2d"}:
        raise ValueError(f"Unknown bounding-box camera mode: {mode}")
    return mode


def instance_mapping(plan):
    """Return stable runtime-label -> scene-object mapping; never infer by class."""
    mapping = {}
    for obj in plan.get("objects", []):
        category = obj.get("category")
        if category not in {"transformer", "switchgear", "capacitor_bank", "reactor"}:
            continue
        labels = obj.get("runtime_labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Unresolvable instance mapping for {obj.get('name')}")
        try:
            unique = {int(value) for value in labels}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unresolvable instance label for {obj.get('name')}") from error
        if len(unique) != 1:
            raise ValueError(f"Unresolvable multiple instance labels for {obj.get('name')}")
        label = unique.pop()
        if label in mapping:
            raise ValueError(f"Instance label collision: {label}")
        mapping[label] = {"object_id": obj["name"], "category": category}
    if not mapping:
        raise ValueError("No target instance mappings")
    return mapping


def _configured_bounds(config):
    origin = config["gazebo_world_origin_m"]
    result = []
    for item in config["obstacles"]:
        x0 = item.get("x_min", item.get("x")) + origin[0]
        x1 = item.get("x_max", item.get("x")) + 1 + origin[0]
        y0 = item.get("y_min", item.get("y")) + origin[1]
        y1 = item.get("y_max", item.get("y")) + 1 + origin[1]
        result.append((item["name"], [x0, x1, y0, y1,
                                      item.get("z_min_m", 0), item["z_max_m"]]))
    return origin, result


def validate_point(point, config, *, role):
    if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
        raise ValueError(f"{role} is not a finite 3D point")
    x, y, z = map(float, point)
    origin, obstacles = _configured_bounds(config)
    # The map boundary itself is legal; configured obstacles are expanded inclusively.
    if not (origin[0] <= x <= origin[0] + config["width"] and
            origin[1] <= y <= origin[1] + config["height"]):
        raise ValueError(f"{role} is outside map bounds")
    for name, (x0, x1, y0, y1, z0, z1) in obstacles:
        c = OBSTACLE_CLEARANCE_M
        if x0-c <= x <= x1+c and y0-c <= y <= y1+c and z0-c <= z <= z1+c:
            raise ValueError(f"{role} touches expanded obstacle: {name}")


def validate_view_pose(view, config, *, actual_carrier=None):
    validate_point(view["camera_position"], config, role="planned camera optical center")
    validate_point(view["position"], config, role="planned carrier reference point")
    if actual_carrier is not None:
        validate_point(actual_carrier, config, role="actual carrier reference point")


def validate_preflight(plan, plan_dir, views):
    expected = (plan.get("annotation_mode"), plan.get("label_mode"),
                plan.get("hierarchy_mode"))
    required = (REQUIRED_ANNOTATION_MODE, REQUIRED_LABEL_MODE, REQUIRED_HIERARCHY_MODE)
    if expected != required:
        raise ValueError(f"Canonical annotation declaration mismatch: {expected!r} != {required!r}")
    world = Path(plan_dir) / "world.sdf"
    actual = annotation_mode_from_world(world)
    if actual != plan["annotation_mode"]:
        raise ValueError(f"Declared/world annotation mode mismatch: {plan['annotation_mode']} != {actual}")
    mapping = instance_mapping(plan)
    config_path = Path(plan_dir) / "obstacles.json"
    config = json.loads(config_path.read_text())
    for view in views:
        validate_view_pose(view, config)
        target = next((v for v in mapping.values() if v["object_id"] == view.get("object_id")), None)
        if view.get("category") in {"transformer", "switchgear", "capacitor_bank", "reactor"} and target is None:
            raise ValueError(f"Planned target instance is not mapped: {view.get('object_id')}")
    return {
        "check_version": CHECK_VERSION,
        "declared_annotation_mode": plan["annotation_mode"],
        "actual_annotation_mode": actual,
        "label_mode": plan["label_mode"],
        "hierarchy_mode": plan["hierarchy_mode"],
        "world_sha256": file_sha256(world),
        "obstacle_clearance_m": OBSTACLE_CLEARANCE_M,
        "instance_mapping": {str(k): v for k, v in sorted(mapping.items())},
    }, config, mapping


def observed_instance_labels(raw_truth):
    items = raw_truth.get("annotated_box", raw_truth.get("annotatedBox", []))
    labels = []
    for item in items:
        try:
            labels.append(int(item["label"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Truth instance label cannot be resolved") from error
    return labels


def target_checks(view, raw_truth, mapping):
    labels = observed_instance_labels(raw_truth)
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f"Truth contains unmapped instance labels: {unknown}")
    expected_labels = [label for label, row in mapping.items()
                       if row["object_id"] == view.get("object_id")]
    if view.get("category") in {"transformer", "switchgear", "capacitor_bank", "reactor"}:
        if len(expected_labels) != 1:
            raise ValueError("Planned target instance mapping is not unique")
        planned_present = expected_labels[0] in labels
    else:
        planned_present = None
    category_present = any(mapping[label]["category"] == view.get("category") for label in labels)
    return {
        "category_present": category_present,
        "planned_instance_present": planned_present,
        "visibility_review": "unknown",
        "observed_instance_labels": labels,
    }
