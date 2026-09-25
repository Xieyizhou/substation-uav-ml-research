"""Data contracts and identities for active semantic inspection."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Any, Mapping


ALLOWED_CLASSES = ("capacitor_bank", "reactor", "switchgear", "transformer")
ALLOWED_TRIGGERS = {
    "stable_detection", "waypoint_reached", "target_completed", "budget_update",
    "safety_replan_active", "safety_replan_cleared",
}
FROZEN_WEIGHTS = {
    "new_class": 4.0, "uninspected": 3.0, "confidence_gap": 2.0,
    "visibility_improvement": 1.5, "path_length": -0.08, "energy": -0.12,
    "risk": -1.0, "repeat_observation": -0.5,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def object_identity(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ActiveInspectionPolicy:
    policy_id: str
    model_artifact_identity: str
    weights: Mapping[str, float]
    stable_confidence: float = .60
    merge_distance_m: float = 2.0
    conflict_distance_m: float = 1.5
    track_timeout_s: float = 15.0
    preferred_standoff_m: float = 5.0
    inspection_confidence: float = .85
    exploration_stride_cells: int = 4
    exploration_completion: float = .90
    mission_budget_s: float = 600.0
    observation_hold_s: float = 2.0
    low_clearance_cells: int = 2
    confirmation_sweep_passes: int = 1
    confirmation_max_route_cells: int = 80

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActiveInspectionPolicy":
        if value.get("runtime_mode") != "active_semantic_inspection":
            raise ValueError("policy runtime_mode must be active_semantic_inspection")
        weights = {key: float(value.get("weights", {}).get(key, math.nan)) for key in FROZEN_WEIGHTS}
        if weights != FROZEN_WEIGHTS:
            raise ValueError("active inspection scoring weights are frozen")
        thresholds, limits = value.get("thresholds", {}), value.get("limits", {})
        return cls(
            policy_id=str(value["policy_id"]), model_artifact_identity=str(value["model_artifact_identity"]),
            weights=weights, stable_confidence=float(thresholds.get("stable_confidence", .60)),
            merge_distance_m=float(thresholds.get("merge_distance_m", 2)),
            conflict_distance_m=float(thresholds.get("conflict_distance_m", 1.5)),
            track_timeout_s=float(thresholds.get("track_timeout_s", 15)),
            preferred_standoff_m=float(thresholds.get("preferred_standoff_m", 5)),
            inspection_confidence=float(thresholds.get("inspection_confidence", .85)),
            exploration_stride_cells=int(limits.get("exploration_stride_cells", 4)),
            exploration_completion=float(limits.get("exploration_completion", .90)),
            mission_budget_s=float(limits.get("mission_budget_s", 600)),
            observation_hold_s=float(limits.get("observation_hold_s", 2)),
            low_clearance_cells=int(limits.get("low_clearance_cells", 2)),
            confirmation_sweep_passes=int(limits.get("confirmation_sweep_passes", 1)),
            confirmation_max_route_cells=int(limits.get("confirmation_max_route_cells", 80)),
        )

    @property
    def artifact_identity(self) -> str:
        return object_identity(asdict(self))


@dataclass
class DeviceTrack:
    target_id: str
    class_name: str
    confidence: float
    east_m: float
    north_m: float
    altitude_m: float
    observation_count: int
    best_confidence: float
    best_depth_m: float
    last_seen_s: float
    source_tracking_ids: list[str] = field(default_factory=list)
    inspected: bool = False
    ambiguous: bool = False


@dataclass(frozen=True)
class RuntimeMap:
    width_cells: int
    height_cells: int
    resolution_m: float
    occupied_cells: frozenset[tuple[int, int]]
    map_identity: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeMap":
        resolution = float(value.get("resolution_m", value.get("cell_size_m", 1)))
        width = int(value.get("width_cells", math.ceil(float(value.get("width_m", value.get("width", 0))) / resolution)))
        height = int(value.get("height_cells", math.ceil(float(value.get("height_m", value.get("height", 0))) / resolution)))
        occupied = {(int(cell[0]), int(cell[1])) for cell in value.get("occupied_cells", [])}
        # Object semantics are deliberately ignored; only explicit geometry is read.
        for obj in value.get("objects", []):
            geometry = obj.get("geometry", {})
            occupied.update((int(c[0]), int(c[1])) for c in geometry.get("occupied_cells", geometry.get("cells", [])))
            footprint = geometry.get("footprint", {})
            x, y = int(footprint.get("x_cell", -1)), int(footprint.get("y_cell", -1))
            w, h = int(footprint.get("width_cells", 0)), int(footprint.get("height_cells", 0))
            occupied.update((cx, cy) for cx in range(x, x + w) for cy in range(y, y + h))
            if "east_m" in obj and "north_m" in obj:
                half_width = float(obj.get("width_m", resolution)) / 2
                half_depth = float(obj.get("depth_m", resolution)) / 2
                west = math.floor((float(obj["east_m"]) - half_width) / resolution)
                east = math.ceil((float(obj["east_m"]) + half_width) / resolution)
                south = math.floor((float(obj["north_m"]) - half_depth) / resolution)
                north = math.ceil((float(obj["north_m"]) + half_depth) / resolution)
                occupied.update(
                    (cx, cy)
                    for cx in range(max(0, west), min(width, east))
                    for cy in range(max(0, south), min(height, north))
                )
        if width <= 0 or height <= 0 or resolution <= 0:
            raise ValueError("map bounds and resolution must be positive")
        view = {"width_cells": width, "height_cells": height, "resolution_m": resolution,
                "occupied_cells": sorted([list(c) for c in occupied])}
        return cls(width, height, resolution, frozenset(occupied), object_identity(view))


