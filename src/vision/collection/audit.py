"""Offline integrity audit for visual collection layouts, routes, and worlds."""

from collections import Counter

from src.ml import EQUIPMENT_CLASSES
from src.vision.collection.layout import (
    build_layout_manifest,
    layout_obstacle_config,
    validate_layout_manifest,
)
from src.vision.collection.plan import validate_collection_plan
from src.vision.collection.route import build_visual_route
from src.vision.collection.world import build_layout_world, validate_world_matches_layout
from src.vision.contracts.protocol import V2_PROTOCOL_ID, protocol_for_plan


def audit_collection_plan(plan):
    validation = validate_collection_plan(plan)
    protocol = protocol_for_plan(plan)
    if protocol.protocol_id != V2_PROTOCOL_ID:
        raise ValueError("collection-audit currently requires visual protocol v2")
    rows_by_layout = {}
    for row in plan["scenarios"]:
        rows_by_layout.setdefault(row["layout_id"], []).append(row)
    category_counts = Counter()
    route_counts = Counter()
    world_count = 0
    for layout_id, rows in sorted(rows_by_layout.items()):
        first = rows[0]
        layout = build_layout_manifest(
            layout_id,
            first["split"],
            first["layout_seed"],
            protocol["randomization"]["configuration"],
        )
        validate_layout_manifest(layout)
        if layout.layout_identity_sha256 != first["layout_identity_sha256"]:
            raise ValueError(f"layout identity mismatch: {layout_id}")
        obstacle_names = {item["name"] for item in layout_obstacle_config(layout)["obstacles"]}
        if obstacle_names != {item.object_id for item in layout.objects}:
            raise ValueError(f"planner/layout mismatch: {layout_id}")
        world = build_layout_world(layout)
        validate_world_matches_layout(world, layout)
        world_count += 1
        category_counts.update(item.visual_category for item in layout.objects)
        for row in rows:
            route = build_visual_route(
                layout,
                row["route_id"],
                row["target_class"],
                camera_heading_offset_deg=protocol["recording"]["camera_heading_offset_deg"],
            )
            if route.route_identity_sha256 != row["route_identity_sha256"]:
                raise ValueError(f"route identity mismatch: {row['scenario_id']}")
            route_counts[row["route_id"]] += 1
    if any(category_counts[name] < 20 for name in EQUIPMENT_CLASSES):
        raise ValueError("v2 aggregate layout inventory is incomplete")
    return {
        **validation,
        "audit_passed": True,
        "layout_count": len(rows_by_layout),
        "world_count": world_count,
        "route_count": sum(route_counts.values()),
        "route_counts": dict(sorted(route_counts.items())),
        "object_category_counts": dict(sorted(category_counts.items())),
    }
