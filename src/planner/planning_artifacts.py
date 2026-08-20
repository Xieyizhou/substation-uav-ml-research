"""Serializable semantic-inspection and height-layer planning workflows."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_routes import build_sandbox_route
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.planner.astar_25d import grid_from_sandbox_map, plan_height_layer_route
from src.planner.semantic_inspection import (
    CameraIntrinsics,
    VehiclePose,
    localize_equipment,
    plan_semantic_inspection,
)


def _read_map(path):
    return SandboxMap.from_record(json.loads(Path(path).read_text()))


def materialize_semantic_inspection(map_path, observation_path, output_path):
    map_path, observation_path = Path(map_path), Path(observation_path)
    payload = json.loads(observation_path.read_text())
    map_value = _read_map(map_path)
    estimate = localize_equipment(
        payload["observation"],
        payload["depth_m"],
        CameraIntrinsics(**payload["camera_intrinsics"]),
        VehiclePose(**payload["vehicle_pose"]),
    )
    plan = plan_semantic_inspection(map_value, estimate)
    result = {
        "semantic_inspection_artifact_schema_version": 1,
        "map_identity_sha256": map_value.map_identity_sha256,
        "map_file_sha256": file_sha256(map_path),
        "observation_file_sha256": file_sha256(observation_path),
        "estimate": asdict(plan.estimate),
        "matched_object_id": plan.matched_object_id,
        "localization_error_m": plan.localization_error_m,
        "mission": asdict(plan.mission),
        "route": plan.route,
        "route_quality": plan.route_quality,
        "semantic_plan_identity_sha256": plan.plan_identity_sha256,
    }
    write_json(output_path, result)
    return result


def materialize_height_layer_plan(
    map_path,
    mission_id,
    output_path,
    *,
    layer_altitudes_m=(1.5, 3.0, 5.0),
):
    map_path = Path(map_path)
    map_value = _read_map(map_path)
    mission = next(
        (item for item in map_value.missions if item.mission_id == mission_id), None
    )
    if mission is None:
        raise ValueError(f"sandbox mission does not exist: {mission_id}")
    route_2d = build_sandbox_route(map_value, mission)
    layers = tuple(float(value) for value in layer_altitudes_m)
    grid = grid_from_sandbox_map(
        map_value,
        layers,
        horizontal_inflation_cells=mission.horizontal_inflation_cells,
    )
    route = plan_height_layer_route(
        grid, (*route_2d.start_cell, 0), (*route_2d.goal_cell, 0)
    )
    record = {
        "height_layer_plan_schema_version": 1,
        "map_identity_sha256": map_value.map_identity_sha256,
        "map_file_sha256": file_sha256(map_path),
        "mission": asdict(mission),
        "layer_altitudes_m": list(layers),
        "blocked_node_count": len(grid.blocked),
        "nodes": [list(node) for node in route.nodes],
        "distance_m": route.distance_m,
        "horizontal_distance_m": route.horizontal_distance_m,
        "vertical_distance_m": route.vertical_distance_m,
        "layer_change_count": route.layer_change_count,
        "two_dimensional_route_identity_sha256": route_2d.route_identity_sha256,
    }
    record["height_layer_plan_identity_sha256"] = object_sha256(record)
    write_json(output_path, record)
    return record
