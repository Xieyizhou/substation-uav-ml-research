"""Frozen 50-recording allocation for visual collection protocol v2."""

from pathlib import Path
import json

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256, write_json
from src.vision.collection.layout import build_layout_manifest
from src.vision.collection.route import build_visual_route


V2_PLAN_SCHEMA_VERSION = 2
V2_SPLITS = ("development", "validation", "blind")
ROLE_BY_SPLIT = {
    "development": "development",
    "validation": "validation",
    "blind": "held_out_test",
}


def _layout_rows(protocol, split, allocation):
    rows = []
    first_seed = int(allocation["seed_first"])
    for layout_index, layout_id in enumerate(allocation["layout_ids"]):
        layout_seed = first_seed + layout_index * len(protocol["routes"])
        layout = build_layout_manifest(
            layout_id,
            split,
            layout_seed,
            protocol["randomization"]["configuration"],
        )
        for route_index, route_spec in enumerate(protocol["routes"]):
            seed = layout_seed + route_index
            route = build_visual_route(
                layout,
                route_spec["route_id"],
                route_spec["target_class"],
                camera_heading_offset_deg=protocol["recording"]["camera_heading_offset_deg"],
            )
            scenario_id = f"{layout_id}-{route_spec['route_id']}-{seed}"
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "recording_id": f"visual-v2-{split}-{layout_id}-{route_spec['route_id']}-{seed}-r01",
                    "split": split,
                    "dataset_role": ROLE_BY_SPLIT[split],
                    "layout_id": layout_id,
                    "layout_seed": layout_seed,
                    "map_id": f"visual-{layout_id}",
                    "target_id": route_spec["target_class"] or "background",
                    "seed": seed,
                    "route_id": route_spec["route_id"],
                    "target_class": route_spec["target_class"],
                    "expected_map_class_inventory": list(EQUIPMENT_CLASSES),
                    "layout_identity_sha256": layout.layout_identity_sha256,
                    "route_identity_sha256": route.route_identity_sha256,
                    "base_scenario_config_hash": object_sha256(
                        {
                            "layout_identity_sha256": layout.layout_identity_sha256,
                            "route_identity_sha256": route.route_identity_sha256,
                            "scenario_seed": seed,
                        }
                    ),
                }
            )
    return rows


def build_v2_collection_plan(protocol):
    rows = []
    for split in V2_SPLITS:
        rows.extend(_layout_rows(protocol, split, protocol["layout_allocation"][split]))
    plan = {
        "collection_plan_schema_version": V2_PLAN_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_identity_sha256": protocol.identity_sha256,
        "scenario_count": len(rows),
        "split_scenario_counts": {split: sum(row["split"] == split for row in rows) for split in V2_SPLITS},
        "scenarios": rows,
    }
    plan["collection_plan_identity_sha256"] = object_sha256(plan)
    return plan


def validate_v2_collection_plan(plan, protocol):
    supplied = plan.get("collection_plan_identity_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "collection_plan_identity_sha256"}
    if supplied != object_sha256(unsigned):
        raise ValueError("visual collection plan identity mismatch")
    if plan != build_v2_collection_plan(protocol):
        raise ValueError("visual v2 collection plan differs from the frozen allocation")
    rows = plan["scenarios"]
    if len(rows) != 50 or len({row["seed"] for row in rows}) != 50:
        raise ValueError("visual v2 collection requires 50 unique scenario seeds")
    layouts_by_split = {
        split: {row["layout_id"] for row in rows if row["split"] == split}
        for split in V2_SPLITS
    }
    if any(layouts_by_split[first] & layouts_by_split[second] for index, first in enumerate(V2_SPLITS) for second in V2_SPLITS[index + 1:]):
        raise ValueError("visual v2 layouts cannot cross splits")
    for layout_id in {row["layout_id"] for row in rows}:
        layout_rows = [row for row in rows if row["layout_id"] == layout_id]
        if len(layout_rows) != 5 or {row["target_class"] for row in layout_rows} != {*EQUIPMENT_CLASSES, None}:
            raise ValueError(f"layout {layout_id} does not contain all frozen routes")
    return {
        "valid": True,
        "protocol_id": protocol.protocol_id,
        "collection_plan_identity_sha256": supplied,
        "scenario_count": len(rows),
        "split_scenario_counts": plan["split_scenario_counts"],
        "layout_count": len({row["layout_id"] for row in rows}),
    }


def write_v2_plan_bundle(output, plan, protocol):
    output = Path(output)
    layouts_root = output.parent / "layouts"
    layout_index = []
    seen = set()
    for row in plan["scenarios"]:
        if row["layout_id"] in seen:
            continue
        seen.add(row["layout_id"])
        layout = build_layout_manifest(
            row["layout_id"],
            row["split"],
            row["layout_seed"],
            protocol["randomization"]["configuration"],
        )
        path = layouts_root / f"{layout.layout_id}.json"
        write_json(path, layout.to_record())
        layout_index.append(
            {
                "layout_id": layout.layout_id,
                "split": layout.split,
                "layout_seed": layout.layout_seed,
                "layout_identity_sha256": layout.layout_identity_sha256,
                "path": str(path.relative_to(output.parent)),
            }
        )
    index = {
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "protocol_identity_sha256": protocol.identity_sha256,
        "layout_count": len(layout_index),
        "layouts": layout_index,
    }
    index["layout_index_identity_sha256"] = object_sha256(index)
    write_json(output.parent / "layout_index.json", index)
    write_json(output, plan)


def validate_v2_plan_bundle(plan_path, plan, protocol):
    root = Path(plan_path).parent
    index_path = root / "layout_index.json"
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    unsigned = {
        key: value
        for key, value in index.items()
        if key != "layout_index_identity_sha256"
    }
    if index.get("layout_index_identity_sha256") != object_sha256(unsigned):
        raise ValueError("visual v2 layout index identity mismatch")
    if (
        index.get("collection_plan_identity_sha256")
        != plan["collection_plan_identity_sha256"]
        or index.get("protocol_identity_sha256") != protocol.identity_sha256
        or index.get("layout_count") != 10
    ):
        raise ValueError("visual v2 layout index differs from plan")
    if len(index.get("layouts", [])) != 10:
        raise ValueError("visual v2 layout index must contain ten layouts")
    rows = {row["layout_id"]: row for row in plan["scenarios"]}
    for item in index.get("layouts", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("visual v2 layout path must stay inside bundle")
        row = rows[item["layout_id"]]
        expected = json.loads(
            json.dumps(
                build_layout_manifest(
                    row["layout_id"],
                    row["split"],
                    row["layout_seed"],
                    protocol["randomization"]["configuration"],
                ).to_record()
            )
        )
        actual = json.loads((root / relative).read_text(encoding="utf-8"))
        if actual != expected or item["layout_identity_sha256"] != expected[
            "layout_identity_sha256"
        ]:
            raise ValueError(f"visual v2 layout bundle mismatch: {row['layout_id']}")
