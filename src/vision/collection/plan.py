"""Frozen multi-scenario visual collection planning and split validation."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256, write_json
from src.ml.domain_randomization import load_ranges, sample_manifest
from src.ml.scenarios import EVALUATION_SEEDS, split_for
from src.maps.map_catalog import map_by_id, project_path
from src.maps.target_catalog import target_by_id
from src.vision.contracts.protocol import (
    PROTOCOL_PATHS,
    V1_PROTOCOL_ID,
    V2_PROTOCOL_ID,
    load_protocol,
    protocol_for_plan,
)


DEFAULT_PROTOCOL = PROTOCOL_PATHS[V1_PROTOCOL_ID]
PLAN_SCHEMA_VERSION = 1
SPLIT_ORDER = ("train", "validation", "test")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing visual collection file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed visual collection JSON: {path}: {error}") from error


def load_collection_protocol(path=DEFAULT_PROTOCOL):
    return load_protocol(path)


def _map_class_inventory(map_id):
    entry = map_by_id(map_id)
    config = _read_json(project_path(entry["obstacle_config"]))
    return sorted(
        {
            item.get("visual_category")
            for item in config.get("obstacles", [])
            if item.get("visual_category") in EQUIPMENT_CLASSES
        },
        key=EQUIPMENT_CLASSES.index,
    )


def _allocation_rows(split, specification):
    seeds = list(
        range(
            int(specification["seed_first"]),
            int(specification["seed_last"]) + 1,
        )
    )
    maps = tuple(specification["maps"])
    targets = tuple(specification["targets"])
    combinations = [(map_id, target_id) for map_id in maps for target_id in targets]
    if not combinations or len(seeds) % len(combinations):
        raise ValueError(f"{split} seeds must evenly cover map-target combinations")
    rows = []
    for index, seed in enumerate(seeds):
        map_id, target_id = combinations[index % len(combinations)]
        entry = map_by_id(map_id)
        target_by_id(entry, target_id)
        expected_split = split_for(map_id, seed)
        if expected_split != split:
            raise ValueError(
                f"{map_id}-{target_id}-{seed} belongs to {expected_split}, not {split}"
            )
        scenario_id = f"{map_id}-{target_id}-{seed}"
        rows.append(
            {
                "scenario_id": scenario_id,
                "recording_id": f"visual-{split}-{scenario_id}-r01",
                "split": split,
                "dataset_role": (
                    "development"
                    if split == "train"
                    else "held_out_test" if split == "test" else "validation"
                ),
                "map_id": map_id,
                "target_id": target_id,
                "seed": seed,
                "route_id": f"{map_id}_{target_id}_fly_round_trip_v1",
                "expected_map_class_inventory": _map_class_inventory(map_id),
            }
        )
    return rows


def visual_randomization_identity(manifest, protocol):
    fields = protocol["randomization"]["applied_visual_fields"]
    return object_sha256({field: manifest[field] for field in fields})


def build_collection_plan(protocol_path=DEFAULT_PROTOCOL):
    protocol = load_collection_protocol(protocol_path)
    if protocol.protocol_id == V2_PROTOCOL_ID:
        from src.vision.collection.v2_plan import build_v2_collection_plan

        return build_v2_collection_plan(protocol)
    randomization_path = project_path(protocol["randomization"]["configuration"])
    randomization = load_ranges(randomization_path)
    rows = []
    for split in SPLIT_ORDER:
        rows.extend(
            _allocation_rows(split, protocol["scenario_allocation"][split])
        )
    for row in rows:
        manifest = sample_manifest(
            randomization,
            map_id=row["map_id"],
            seed=row["seed"],
        )
        row["base_scenario_config_hash"] = visual_randomization_identity(
            manifest, protocol
        )
    plan = {
        "collection_plan_schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "scenario_count": len(rows),
        "split_scenario_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in SPLIT_ORDER
        },
        "scenarios": rows,
    }
    plan["collection_plan_identity_sha256"] = object_sha256(plan)
    return plan


def validate_collection_plan(plan, *, protocol_path=None):
    protocol = (
        protocol_for_plan(plan)
        if protocol_path is None
        else load_collection_protocol(protocol_path)
    )
    if protocol.protocol_id == V2_PROTOCOL_ID:
        from src.vision.collection.v2_plan import validate_v2_collection_plan

        return validate_v2_collection_plan(plan, protocol)
    supplied = plan.get("collection_plan_identity_sha256")
    unsigned = {
        key: value
        for key, value in plan.items()
        if key != "collection_plan_identity_sha256"
    }
    if supplied != object_sha256(unsigned):
        raise ValueError("visual collection plan identity mismatch")
    expected = build_collection_plan(protocol_path)
    if plan != expected:
        raise ValueError("visual collection plan differs from the frozen allocation")
    rows = plan.get("scenarios") or []
    seeds = [int(row["seed"]) for row in rows]
    scenario_ids = [row["scenario_id"] for row in rows]
    recording_ids = [row["recording_id"] for row in rows]
    if len(rows) != 60 or len(set(seeds)) != 60:
        raise ValueError("visual collection requires 60 unique scenario seeds")
    if EVALUATION_SEEDS.intersection(seeds):
        raise ValueError("formal evaluation seeds cannot enter visual datasets")
    if len(set(scenario_ids)) != len(rows) or len(set(recording_ids)) != len(rows):
        raise ValueError("visual collection scenario and recording IDs must be unique")
    for split in SPLIT_ORDER:
        inventory = {
            class_name
            for row in rows
            if row["split"] == split
            for class_name in row["expected_map_class_inventory"]
        }
        if inventory != set(EQUIPMENT_CLASSES):
            raise ValueError(f"{split} map inventory does not cover all classes")
    return {
        "valid": True,
        "protocol_id": protocol["protocol_id"],
        "collection_plan_identity_sha256": supplied,
        "scenario_count": len(rows),
        "split_scenario_counts": plan["split_scenario_counts"],
    }


def write_collection_plan(output, protocol_path=DEFAULT_PROTOCOL):
    plan = build_collection_plan(protocol_path)
    validate_collection_plan(plan, protocol_path=protocol_path)
    if plan["protocol_id"] == V2_PROTOCOL_ID:
        from src.vision.collection.v2_plan import write_v2_plan_bundle

        write_v2_plan_bundle(output, plan, load_collection_protocol(protocol_path))
    else:
        write_json(output, plan)
    return plan


def load_collection_plan(path, *, protocol_path=None):
    plan = _read_json(path)
    validate_collection_plan(plan, protocol_path=protocol_path)
    if plan.get("protocol_id") == V2_PROTOCOL_ID:
        from src.vision.collection.v2_plan import validate_v2_plan_bundle

        validate_v2_plan_bundle(path, plan, protocol_for_plan(plan))
    return plan


def collection_status(plan, recordings_root):
    from src.vision.collection.receipt import (
        RECEIPT_NAME,
        collection_validation_receipt_is_current,
    )

    recordings_root = Path(recordings_root)
    states = {
        "complete": 0,
        "invalid_receipt": 0,
        "unvalidated": 0,
        "failed": 0,
        "recording": 0,
        "partial": 0,
        "missing": 0,
    }
    next_scenario = None
    for row in plan["scenarios"]:
        root = recordings_root / row["recording_id"]
        identity_exists = (root / "identity/recording_identity.json").is_file()
        if (root / RECEIPT_NAME).is_file():
            if collection_validation_receipt_is_current(root, row, plan):
                state = "complete"
            else:
                state = "invalid_receipt"
        elif (root / "summary.json").is_file() and identity_exists:
            try:
                metadata = _read_json(root / "metadata.json")
            except (FileNotFoundError, ValueError):
                metadata = {}
            lifecycle = metadata.get("flight_lifecycle") or {}
            if (
                metadata.get("recording_state") != "complete"
                or lifecycle.get("event_type") == "mission_failed"
            ):
                state = "failed"
            else:
                state = "unvalidated"
        elif (root / "live_status.json").is_file():
            state = "recording"
        elif root.exists():
            state = "partial"
        else:
            state = "missing"
        states[state] += 1
        if next_scenario is None and state != "complete":
            next_scenario = {**row, "recording_directory": str(root), "state": state}
    return {
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "recording_states": states,
        "next_scenario": next_scenario,
    }
