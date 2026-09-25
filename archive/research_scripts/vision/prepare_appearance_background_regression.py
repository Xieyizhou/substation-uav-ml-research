"""Freeze development-only appearance and background variation holdouts."""
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

BASE = ROOT / "data/research/ml_training_recovery_v1/appearance-background-v1"
SOURCE = ROOT / "data/research/canonical_views_v1/complex-occlusion-filtered-plan-v1/plan.json"
CONFIG = ROOT / "config/maps/substation_complex.json"
SPEC = {"transformer": 3, "switchgear": 3, "capacitor_bank": 3, "reactor": 3}
VARIANTS = {"appearance": 181, "background": 191}


def known_view_ids():
    known = set()
    for receipt_path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(receipt_path) or "appearance-background-v1" in str(receipt_path):
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        known.update(row.get("view_id") for row in receipt.get("views", []) if row.get("view_id"))
    return known


def choose_rows(rows, count):
    rows = sorted(rows, key=lambda row: (-row["distance"], row["object_id"], row["bearing"], row["height"], row["offset"], row["view_id"]))
    selected, used_bins = [], set()
    for row in rows:
        key = (row["object_id"], row["bearing"] // 30)
        if key in used_bins:
            continue
        selected.append(row); used_bins.add(key)
        if len(selected) == count:
            return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(f"Not enough candidates: need {count}, have {len(rows)}")


def set_color(material, values):
    text = " ".join(f"{value:.3f}" for value in values) + " 1"
    for tag in ("ambient", "diffuse"):
        node = material.find(tag)
        if node is not None:
            node.text = text


def mutate_world(world, variant, objects):
    tree = ET.parse(world)
    root = tree.getroot(); world_node = root.find("world")
    if world_node is None:
        raise ValueError(f"Missing world element: {world}")
    changed = []
    categories = {obj["name"]: obj["category"] for obj in objects}
    if variant == "appearance":
        colors = {
            "transformer": (0.34, 0.20, 0.12),
            "switchgear": (0.42, 0.28, 0.08),
            "capacitor_bank": (0.22, 0.42, 0.24),
            "reactor": (0.42, 0.16, 0.22),
        }
        for model in world_node.iter("model"):
            category = categories.get(model.get("name"))
            if category not in colors:
                continue
            for visual in model.iter("visual"):
                if visual.get("name") not in {"body", "front_panel", "reactor"}:
                    continue
                for material in visual.findall("material"):
                    set_color(material, colors[category])
                    changed.append(f"{model.get('name')}:{visual.get('name')}")
    elif variant == "background":
        scene = world_node.find("scene")
        if scene is None:
            raise ValueError("Missing scene")
        ambient = scene.find("ambient"); background = scene.find("background")
        if ambient is None or background is None:
            raise ValueError("Missing scene lighting")
        ambient.text = "0.38 0.42 0.48 1"; background.text = "0.16 0.22 0.31 1"
        light = world_node.find("light[@name='sun']")
        diffuse = light.find("diffuse") if light is not None else None
        if diffuse is None:
            raise ValueError("Missing sun diffuse")
        diffuse.text = "0.58 0.66 0.78 1"
        changed.extend(["scene/ambient", "scene/background", "sun/diffuse"])
        for model in world_node.iter("model"):
            if model.get("name") not in {"substation_floor", "substation_floor_grid"}:
                continue
            for material in model.iter("material"):
                set_color(material, (0.27, 0.23, 0.19) if model.get("name") == "substation_floor" else (0.10, 0.11, 0.13))
                changed.append(f"{model.get('name')}:material")
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if not changed:
        raise ValueError(f"Variant made no changes: {variant}")
    ET.indent(tree, space="  "); tree.write(world, encoding="utf-8", xml_declaration=True)
    return changed


def copy_world(source_path, target_dir, variant):
    source = read_record(source_path); target_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in source["files"].items():
        source_file = source_path.parent / name
        if file_sha256(source_file) != digest:
            raise ValueError(f"Changed source artifact: {source_file}")
        (target_dir / name).write_bytes(source_file.read_bytes())
    world = target_dir / "world.sdf"
    changed = mutate_world(world, variant, source["objects"])
    tree = ET.parse(world)
    sensors = [sensor for sensor in tree.iter("sensor") if sensor.get("type") == "boundingbox_camera"]
    if len(sensors) != 1:
        raise ValueError("Expected one bounding-box camera")
    box_type = sensors[0].find("camera/box_type")
    if box_type is None:
        raise ValueError("Missing camera box_type")
    box_type.text = "full_2d"
    ET.indent(tree, space="  "); tree.write(world, encoding="utf-8", xml_declaration=True)
    if [sensor for sensor in ET.parse(world).iter("sensor") if sensor.get("type") == "boundingbox_camera"][0].findtext("camera/box_type") != "full_2d":
        raise ValueError("Bounding-box camera mode did not persist")
    return source, world, changed


def prepare():
    manifest_path = BASE / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    source = read_record(SOURCE); config = json.loads(CONFIG.read_text()); known = known_view_ids(); runs = []
    for variant, round_index in VARIANTS.items():
        candidates = target_candidates("complex", source["objects"], config["gazebo_world_origin_m"][:2], (config["width"], config["height"]), round_index)
        selected = []
        for category, count in SPEC.items():
            options = [row for row in candidates if row["category"] == category and row["view_id"] not in known]
            selected.extend(choose_rows(options, count))
        if len({row["view_id"] for row in selected}) != 12:
            raise ValueError(f"Duplicate selected view ids: {variant}")
        plan_dir = BASE / variant / "plan"; source_copy, world, changed = copy_world(SOURCE, plan_dir, variant)
        plan = {key: copy.deepcopy(value) for key, value in source_copy.items() if key != "identity"}
        plan.update(
            annotation_mode="full_2d", calibration_views=selected, pilot_views=[], diagnostic_only=True,
            diagnostic_allow_expected_absence=False, diagnostic_require_expected_presence=True,
            training_admitted=False, automatic_training=False,
            diagnostic_purpose="appearance_background_regression_holdout", variant=variant,
            variant_changes=changed, source_plan_path=str(SOURCE), source_plan_sha256=file_sha256(SOURCE),
            expansion_round=round_index, excluded_known_view_count=len(known),
        )
        plan["files"]["world.sdf"] = file_sha256(world)
        written = write_record(plan_dir / "plan.json", plan)
        runs.append({
            "variant": variant, "map_id": "complex", "plan_path": str(plan_dir / "plan.json"),
            "plan_identity": written["identity"], "selected": [{key: row[key] for key in ("view_id", "category", "object_id", "bearing", "distance", "height", "offset")} for row in selected],
            "candidate_count": len(candidates), "training_admitted": False, "promotable": False,
        })
    manifest = {
        "status": "prepared", "purpose": "appearance_background_regression_holdout", "runs": runs,
        "selection_rule": "Two independent complex-scene variants with unchanged geometry and full_2d labels; 3 frames per target class per variant, far poses preferred.",
        "training_admitted": False, "promotable": False,
    }
    manifest["identity"] = object_sha256(manifest); BASE.mkdir(parents=True, exist_ok=True); write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False)); return manifest


if __name__ == "__main__":
    prepare()
