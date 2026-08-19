"""Deterministic routes for user-authored sandbox maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from src.maps.sandbox_contracts import SandboxMap, SandboxMission
from src.maps.sandbox_geometry import object_cells, point_cell
from src.ml.artifacts import object_sha256
from src.planner.astar_grid import astar, simplify_grid_path
from src.planner.obstacle_config import inflate_cells


@dataclass(frozen=True)
class SandboxRouteWaypoint:
    waypoint_id: str
    mission_phase: str
    east_m: float
    north_m: float
    altitude_m: float
    yaw_deg: float
    hold_s: float = 0.0


@dataclass(frozen=True)
class SandboxRoute:
    route_id: str
    map_identity_sha256: str
    mission_id: str
    mission_type: str
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    grid_path: tuple[tuple[int, int], ...]
    simplified_path: tuple[tuple[int, int], ...]
    return_grid_path: tuple[tuple[int, int], ...]
    waypoints: tuple[SandboxRouteWaypoint, ...]
    estimated_duration_s: float
    sandbox_route_schema_version: int = 1

    @property
    def route_identity_sha256(self):
        return object_sha256(asdict(self))

    def to_record(self):
        return {**asdict(self), "route_identity_sha256": self.route_identity_sha256}


def blocking_cells(map_value: SandboxMap, mission: SandboxMission):
    width, height = int(map_value.width_m), int(map_value.height_m)
    raw = set()
    for item in map_value.objects:
        if mission.altitude_m + 0.3 >= 0.0 and mission.altitude_m - 0.3 <= item.height_m:
            raw.update(object_cells(item, width, height))
    protected = {point_cell(map_value.start_east_m, map_value.start_north_m)}
    inflated = inflate_cells(raw, width, height, mission.horizontal_inflation_cells)
    return raw, inflated - protected


def _path(start, goal, blocked, width, height):
    if start in blocked or goal in blocked:
        raise ValueError(f"route endpoint is blocked: {goal}")
    return tuple(astar(start, goal, blocked, width, height))


def _yaw(east, north, target_east, target_north):
    return math.degrees(math.atan2(target_east - east, target_north - north)) % 360.0


def _point(target_east, target_north, bearing_deg, distance_m):
    angle = math.radians(bearing_deg)
    return (
        target_east + math.sin(angle) * distance_m,
        target_north + math.cos(angle) * distance_m,
    )


def _inside(point, width, height, margin=0.5):
    return margin <= point[0] <= width - margin and margin <= point[1] <= height - margin


def _inspection_points(map_value, mission, blocked):
    target = next(item for item in map_value.objects if item.object_id == mission.target_object_id)
    if target.label_role != "target":
        raise ValueError("inspection target must be a labelled equipment object")
    start = point_cell(map_value.start_east_m, map_value.start_north_m)
    width, height = int(map_value.width_m), int(map_value.height_m)
    distances = (8.0, 5.0, 3.5, 5.0, 3.5)
    for bearing in range(0, 360, 15):
        bearings = (bearing, bearing, bearing, (bearing + 90) % 360, (bearing + 90) % 360)
        points = tuple(
            _point(target.east_m, target.north_m, angle, distance)
            for angle, distance in zip(bearings, distances)
        )
        if not all(_inside(point, width, height) for point in points):
            continue
        current, paths = start, []
        try:
            for point in points:
                destination = point_cell(*point)
                path = _path(current, destination, blocked, width, height)
                paths.append(path)
                current = destination
            return target, points, tuple(paths)
        except ValueError:
            continue
    raise ValueError(f"no reachable inspection route for {target.object_id}")


def _basic_route(map_value, mission, blocked):
    width, height = int(map_value.width_m), int(map_value.height_m)
    start = point_cell(map_value.start_east_m, map_value.start_north_m)
    goal = point_cell(mission.goal_east_m, mission.goal_north_m)
    path = _path(start, goal, blocked, width, height)
    simplified = tuple(simplify_grid_path(path))
    waypoints = tuple(
        SandboxRouteWaypoint(
            f"waypoint_{index:03d}", "outbound", cell[0] + 0.5,
            cell[1] + 0.5, mission.altitude_m, 0.0,
        )
        for index, cell in enumerate(simplified[1:], start=1)
    )
    returning = tuple(reversed(path)) if mission.mission_type == "round_trip" else ()
    distance = (len(path) - 1 + max(0, len(returning) - 1)) * map_value.resolution_m
    return start, goal, path, simplified, returning, waypoints, distance


def build_sandbox_route(map_value: SandboxMap, mission: SandboxMission):
    raw, blocked = blocking_cells(map_value, mission)
    del raw
    if mission.mission_type != "equipment_inspection":
        values = _basic_route(map_value, mission, blocked)
    else:
        target, points, paths = _inspection_points(map_value, mission, blocked)
        phases = ("cruise_distant", "approach", "close_inspection", "target_transition", "close_inspection")
        holds = (2.0, 2.0, 2.5, 2.0, 2.5)
        waypoints = tuple(
            SandboxRouteWaypoint(
                f"inspection_{index:02d}", phase, *point, mission.altitude_m,
                _yaw(*point, target.east_m, target.north_m), hold,
            )
            for index, (point, phase, hold) in enumerate(zip(points, phases, holds), start=1)
        )
        full_path = tuple(cell for index, path in enumerate(paths) for cell in path[(1 if index else 0):])
        start = point_cell(map_value.start_east_m, map_value.start_north_m)
        goal = point_cell(*points[-1])
        returning = _path(goal, start, blocked, int(map_value.width_m), int(map_value.height_m))
        distance = (len(full_path) + len(returning) - 2) * map_value.resolution_m
        values = start, goal, full_path, tuple(point_cell(*point) for point in points), returning, waypoints, distance
    start, goal, path, simplified, returning, waypoints, distance = values
    hold_time = sum(item.hold_s for item in waypoints)
    estimated = distance / mission.speed_m_s + hold_time + 45.0
    return SandboxRoute(
        f"{map_value.map_id}-{mission.mission_id}-v1", map_value.map_identity_sha256,
        mission.mission_id, mission.mission_type, start, goal, path, simplified,
        returning, waypoints, round(estimated, 3),
    )
