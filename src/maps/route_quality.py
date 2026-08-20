"""Deterministic quality gates for sandbox map routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from src.maps.sandbox_contracts import SandboxMap, SandboxMission
from src.maps.sandbox_routes import SandboxRoute, blocking_cells
from src.ml.artifacts import object_sha256


MISSION_BUDGET_S = 600.0
MAX_DETOUR_RATIO = 4.0
MAX_BOUNDARY_EXPOSURE = 0.70
MIN_INTERIOR_COVERAGE = 0.20
MIN_PATH_CELLS = 4
MIN_EFFECTIVE_DISTANCE_M = 3.0


@dataclass(frozen=True)
class RouteQualityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class RouteQualityReport:
    route_identity_sha256: str
    mission_id: str
    mission_type: str
    accepted: bool
    path_cell_count: int
    return_path_cell_count: int
    route_length_m: float
    direct_distance_m: float
    detour_ratio: float | None
    minimum_clearance_m: float | None
    required_clearance_m: float
    boundary_exposure_ratio: float
    interior_coverage_ratio: float
    turn_count: int
    simplified_cell_count: int
    simplification_ratio: float
    estimated_duration_s: float
    mission_budget_s: float
    issues: tuple[RouteQualityIssue, ...]
    route_quality_schema_version: int = 1

    @property
    def route_quality_identity_sha256(self):
        return object_sha256(asdict(self))

    def to_record(self):
        return {
            **asdict(self),
            "route_quality_identity_sha256": self.route_quality_identity_sha256,
        }


def _path_length(path, resolution_m):
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1]) * resolution_m
        for first, second in zip(path, path[1:])
    )


def _turn_count(path):
    directions = [
        (second[0] - first[0], second[1] - first[1])
        for first, second in zip(path, path[1:])
    ]
    return sum(first != second for first, second in zip(directions, directions[1:]))


def _minimum_clearance(path, raw_cells, resolution_m):
    if not path or not raw_cells:
        return None
    return min(
        math.hypot(cell[0] - obstacle[0], cell[1] - obstacle[1]) * resolution_m
        for cell in path
        for obstacle in raw_cells
    )


def _route_cells(route):
    if not route.return_grid_path:
        return tuple(route.grid_path)
    return tuple(route.grid_path) + tuple(route.return_grid_path[1:])


def evaluate_route_quality(
    map_value: SandboxMap,
    mission: SandboxMission,
    route: SandboxRoute,
):
    path = tuple(route.grid_path)
    complete_path = _route_cells(route)
    resolution = map_value.resolution_m
    route_length = _path_length(complete_path, resolution)
    direct_distance = math.hypot(
        route.goal_cell[0] - route.start_cell[0],
        route.goal_cell[1] - route.start_cell[1],
    ) * resolution
    outbound_length = _path_length(path, resolution)
    detour = outbound_length / direct_distance if direct_distance > 0.0 else None
    raw_cells, _ = blocking_cells(map_value, mission)
    clearance = _minimum_clearance(complete_path, raw_cells, resolution)
    required_clearance = mission.horizontal_inflation_cells * resolution
    width, height = int(map_value.width_m), int(map_value.height_m)
    boundary_count = sum(
        cell[0] in {0, width - 1} or cell[1] in {0, height - 1}
        for cell in path
    )
    interior_count = sum(
        2 <= cell[0] <= width - 3 and 2 <= cell[1] <= height - 3
        for cell in path
    )
    boundary_ratio = boundary_count / len(path) if path else 1.0
    interior_ratio = interior_count / len(path) if path else 0.0
    simplification_ratio = len(route.simplified_path) / len(path) if path else 0.0
    issues = []
    if route.start_cell == route.goal_cell:
        issues.append(RouteQualityIssue(
            "route_same_cell", "Route start and goal resolve to the same grid cell."
        ))
    if len(path) < MIN_PATH_CELLS:
        issues.append(RouteQualityIssue(
            "route_too_few_cells", f"Route has fewer than {MIN_PATH_CELLS} cells."
        ))
    if outbound_length < MIN_EFFECTIVE_DISTANCE_M:
        issues.append(RouteQualityIssue(
            "route_too_short",
            f"Route effective distance is below {MIN_EFFECTIVE_DISTANCE_M:g} m.",
        ))
    if mission.mission_type == "round_trip" and (
        not route.return_grid_path
        or route.return_grid_path[0] != route.goal_cell
        or route.return_grid_path[-1] != route.start_cell
    ):
        issues.append(RouteQualityIssue(
            "return_route_invalid", "Round-trip mission has no valid return route."
        ))
    if clearance is not None and clearance < required_clearance:
        issues.append(RouteQualityIssue(
            "route_clearance_low",
            "Route minimum clearance is below the configured safety inflation distance.",
        ))
    if boundary_ratio > MAX_BOUNDARY_EXPOSURE:
        issues.append(RouteQualityIssue(
            "route_boundary_exposure_high",
            f"Route boundary exposure exceeds {MAX_BOUNDARY_EXPOSURE:.0%}.",
        ))
    if interior_ratio < MIN_INTERIOR_COVERAGE:
        issues.append(RouteQualityIssue(
            "route_interior_coverage_low",
            f"Route interior coverage is below {MIN_INTERIOR_COVERAGE:.0%}.",
        ))
    if detour is not None and detour > MAX_DETOUR_RATIO:
        issues.append(RouteQualityIssue(
            "route_detour_excessive",
            f"Route detour ratio exceeds {MAX_DETOUR_RATIO:.1f}.",
        ))
    if route.estimated_duration_s > MISSION_BUDGET_S:
        issues.append(RouteQualityIssue(
            "route_duration_excessive",
            f"Estimated route duration exceeds {MISSION_BUDGET_S:g} seconds.",
        ))
    return RouteQualityReport(
        route.route_identity_sha256,
        mission.mission_id,
        mission.mission_type,
        not issues,
        len(path),
        len(route.return_grid_path),
        round(route_length, 3),
        round(direct_distance, 3),
        None if detour is None else round(detour, 6),
        None if clearance is None else round(clearance, 3),
        round(required_clearance, 3),
        round(boundary_ratio, 6),
        round(interior_ratio, 6),
        _turn_count(path),
        len(route.simplified_path),
        round(simplification_ratio, 6),
        route.estimated_duration_s,
        MISSION_BUDGET_S,
        tuple(issues),
    )
