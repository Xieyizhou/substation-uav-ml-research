"""Materialize immutable, identity-bound sandbox map revisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_flight import visual_route_for_sandbox
from src.maps.sandbox_geometry import object_cells, point_cell
from src.maps.sandbox_preview import preview_png, preview_record, preview_svg
from src.maps.sandbox_routes import blocking_cells
from src.maps.sandbox_validation import validate_sandbox_map
from src.maps.sandbox_world import world_bytes
from src.ml.artifacts import object_sha256


def _sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


@dataclass(frozen=True)
class SandboxMapRevision:
    map_id: str
    map_identity_sha256: str
    validation_identity_sha256: str
    artifact_sha256: dict[str, str]
    route_identity_sha256: tuple[str, ...]
    sandbox_map_revision_schema_version: int = 1

    @property
    def revision_identity_sha256(self):
        return object_sha256(asdict(self))

    def to_record(self):
        return {**asdict(self), "revision_identity_sha256": self.revision_identity_sha256}


def obstacle_record(map_value: SandboxMap, mission=None):
    selected = mission or (map_value.missions[0] if map_value.missions else None)
    raw, blocked = (set(), set()) if selected is None else blocking_cells(map_value, selected)
    goal = (
        point_cell(selected.goal_east_m, selected.goal_north_m)
        if selected is not None and selected.goal_east_m is not None
        else point_cell(map_value.start_east_m, map_value.start_north_m)
    )
    return {
        "map_name": f"sandbox_{map_value.map_id}_v1",
        "width": int(map_value.width_m),
        "height": int(map_value.height_m),
        "resolution_m": map_value.resolution_m,
        "start_cell": list(point_cell(map_value.start_east_m, map_value.start_north_m)),
        "goal_cell": list(goal),
        "gazebo_world_origin_m": [-map_value.start_east_m, -map_value.start_north_m, 0.0],
        "horizontal_inflation_cells": 0 if selected is None else selected.horizontal_inflation_cells,
        "altitude_m": 1.5 if selected is None else selected.altitude_m,
        "obstacles": [
            {
                "name": item.object_id,
                "type": "cells",
                "cells": [list(cell) for cell in sorted(object_cells(item, int(map_value.width_m), int(map_value.height_m)))],
                "z_min_m": 0.0,
                "z_max_m": item.height_m,
                "visual_category": item.asset_id,
            }
            for item in map_value.objects
        ],
        "raw_obstacle_cells": [list(cell) for cell in sorted(raw)],
        "inflated_blocking_cells": [list(cell) for cell in sorted(blocked)],
    }


def _artifacts(map_value, report, routes):
    route_by_id = {route.mission_id: route for route in routes}
    artifacts = {
        "map.json": _json_bytes(map_value.to_record()),
        "world.sdf": world_bytes(map_value),
        "obstacles.json": _json_bytes(obstacle_record(map_value)),
        "preview.json": _json_bytes(preview_record(map_value, routes)),
        "preview.svg": preview_svg(map_value, routes).encode("utf-8"),
        "preview.png": preview_png(map_value, routes),
        "validation.json": _json_bytes(report.to_record()),
        **{
            f"routes/{route.mission_id}.json": _json_bytes(route.to_record())
            for route in routes
        },
    }
    for mission in map_value.missions:
        route = route_by_id[mission.mission_id]
        artifacts[f"routes/{mission.mission_id}.obstacles.json"] = _json_bytes(
            obstacle_record(map_value, mission)
        )
        if mission.mission_type == "equipment_inspection":
            artifacts[f"routes/{mission.mission_id}.visual.json"] = _json_bytes(
                visual_route_for_sandbox(map_value, mission, route).to_record()
            )
    return artifacts


def materialize_revision(map_value: SandboxMap, revisions_root):
    report, routes = validate_sandbox_map(map_value)
    if not report.valid:
        messages = "; ".join(issue.message for issue in report.issues if issue.severity == "error")
        raise ValueError(f"sandbox map validation failed: {messages}")
    artifacts = _artifacts(map_value, report, routes)
    hashes = {name: _sha256(data) for name, data in sorted(artifacts.items())}
    revision = SandboxMapRevision(
        map_value.map_id, map_value.map_identity_sha256,
        report.validation_identity_sha256, hashes,
        tuple(route.route_identity_sha256 for route in routes),
    )
    root = Path(revisions_root) / map_value.map_id / revision.revision_identity_sha256
    if root.exists():
        existing = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        if existing.get("revision_identity_sha256") != revision.revision_identity_sha256:
            raise ValueError("existing sandbox map revision identity differs")
        return root, revision
    root.mkdir(parents=True)
    for relative, data in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    identity = revision.to_record()
    identity["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (root / "identity.json").write_bytes(_json_bytes(identity))
    return root, revision
