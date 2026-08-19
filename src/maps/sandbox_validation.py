"""Validation reports for sandbox map drafts and revisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_geometry import object_inside_map, objects_overlap, point_cell
from src.maps.sandbox_routes import blocking_cells, build_sandbox_route
from src.ml.artifacts import object_sha256


@dataclass(frozen=True)
class MapValidationIssue:
    code: str
    message: str
    severity: str = "error"
    object_id: str | None = None
    mission_id: str | None = None


@dataclass(frozen=True)
class MapValidationReport:
    map_identity_sha256: str
    valid: bool
    checks_passed: int
    checks_total: int
    issues: tuple[MapValidationIssue, ...]
    route_identities: tuple[str, ...]
    sandbox_map_validation_schema_version: int = 1

    @property
    def validation_identity_sha256(self):
        return object_sha256(asdict(self))

    def to_record(self):
        return {**asdict(self), "validation_identity_sha256": self.validation_identity_sha256}


def validate_sandbox_map(map_value: SandboxMap):
    issues = []
    checks_total = 4 + len(map_value.missions)
    if not (0.0 <= map_value.start_east_m < map_value.width_m and 0.0 <= map_value.start_north_m < map_value.height_m):
        issues.append(MapValidationIssue("start_out_of_bounds", "The start position is outside the map."))
    outside = [item for item in map_value.objects if not object_inside_map(item, map_value.width_m, map_value.height_m)]
    issues.extend(MapValidationIssue("object_out_of_bounds", "Object extends outside the map.", object_id=item.object_id) for item in outside)
    for index, first in enumerate(map_value.objects):
        for second in map_value.objects[index + 1:]:
            if objects_overlap(first, second):
                issues.append(MapValidationIssue("object_overlap", f"Objects {first.object_id} and {second.object_id} overlap.", object_id=first.object_id))
    routes = []
    for mission in map_value.missions:
        try:
            _, blocked = blocking_cells(map_value, mission)
            if point_cell(map_value.start_east_m, map_value.start_north_m) in blocked:
                raise ValueError("start position is blocked")
            route = build_sandbox_route(map_value, mission)
            routes.append(route)
            if route.estimated_duration_s > 600.0:
                issues.append(MapValidationIssue("route_duration_warning", "Estimated route duration exceeds 600 seconds.", "warning", mission_id=mission.mission_id))
        except (ValueError, KeyError) as error:
            issues.append(MapValidationIssue("route_unreachable", str(error), mission_id=mission.mission_id))
    errors = [issue for issue in issues if issue.severity == "error"]
    checks_passed = max(0, checks_total - len(errors))
    return MapValidationReport(
        map_value.map_identity_sha256, not errors, checks_passed, checks_total,
        tuple(issues), tuple(route.route_identity_sha256 for route in routes),
    ), tuple(routes)
