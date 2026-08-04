"""Deterministic equipment-centered observation routes for visual collection v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from src.ml.artifacts import object_sha256
from src.planner.astar_grid import astar, simplify_grid_path
from src.planner.obstacle_config import build_obstacle_map
from src.vision.collection.layout import LayoutManifest, layout_obstacle_config


ROUTE_SCHEMA_VERSION = 1
PHASES = ("cruise_distant", "approach", "close_inspection", "target_transition")


@dataclass(frozen=True)
class ObservationWaypoint:
    waypoint_id: str
    mission_phase: str
    east_m: float
    north_m: float
    altitude_m: float
    yaw_deg: float
    hold_s: float
    expected_truth: str
    transit_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class VisualRoute:
    route_id: str
    layout_id: str
    target_class: str | None
    target_object_id: str | None
    start_cell: tuple[int, int]
    waypoints: tuple[ObservationWaypoint, ...]
    route_schema_version: int = ROUTE_SCHEMA_VERSION

    def identity_record(self):
        return asdict(self)

    @property
    def route_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "route_identity_sha256": self.route_identity_sha256}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("route_identity_sha256", None)
        waypoints = []
        for item in values.pop("waypoints"):
            item = dict(item)
            item["transit_cells"] = tuple(tuple(cell) for cell in item["transit_cells"])
            waypoints.append(ObservationWaypoint(**item))
        values["start_cell"] = tuple(values["start_cell"])
        route = cls(waypoints=tuple(waypoints), **values)
        if supplied != route.route_identity_sha256:
            raise ValueError("visual route identity mismatch")
        return route


def _yaw_to_target(east, north, target_east, target_north):
    return math.degrees(math.atan2(target_east - east, target_north - north)) % 360.0


def _point(target, bearing_deg, distance_m):
    angle = math.radians(bearing_deg)
    return target.east_m + math.sin(angle) * distance_m, target.north_m + math.cos(angle) * distance_m


def _cell(east, north):
    return int(math.floor(east)), int(math.floor(north))


def _reachable_sequence(manifest, points):
    source = layout_obstacle_config(manifest)
    config = build_obstacle_map(source)
    current = tuple(source["start_cell"])
    paths = []
    for east, north in points:
        destination = _cell(east, north)
        if destination in config["inflated_blocking_cells"]:
            return None
        try:
            path = astar(current, destination, config["inflated_blocking_cells"], source["width"], source["height"])
        except ValueError:
            return None
        paths.append(tuple(simplify_grid_path(path)[1:]))
        current = destination
    return tuple(paths)


def _target_waypoints(manifest, target):
    rotations = tuple(float(value) for value in range(0, 360, 45))
    start = (manifest.layout_seed - 3001) % len(rotations)
    for offset in range(len(rotations)):
        first_bearing = rotations[(start + offset) % len(rotations)]
        second_bearing = (first_bearing + 90.0) % 360.0
        distant = _point(target, first_bearing, 12.0)
        approach = _point(target, first_bearing, 5.0)
        close_first = _point(target, first_bearing, 3.5)
        transition = _point(target, second_bearing, 5.0)
        close_second = _point(target, second_bearing, 3.5)
        points = (distant, approach, close_first, transition, close_second)
        if any(east <= 11.0 for east, _ in points):
            continue
        paths = _reachable_sequence(manifest, points)
        if paths is None:
            continue
        waypoints = [
            ObservationWaypoint("distant", PHASES[0], *distant, 1.5, _yaw_to_target(*distant, target.east_m, target.north_m), 5.0, "labelled_target", paths[0]),
            ObservationWaypoint("approach", PHASES[1], *approach, 1.5, _yaw_to_target(*approach, target.east_m, target.north_m), 2.0, "labelled_target", paths[1]),
        ]
        for index, yaw_offset in enumerate((-30.0, 0.0, 30.0), start=1):
            yaw = (_yaw_to_target(*close_first, target.east_m, target.north_m) + yaw_offset) % 360.0
            waypoints.append(ObservationWaypoint(f"close_a_{index}", PHASES[2], *close_first, 1.5, yaw, 1.5, "labelled_target", paths[2] if index == 1 else ()))
        waypoints.append(ObservationWaypoint("transition", PHASES[3], *transition, 1.5, _yaw_to_target(*transition, target.east_m, target.north_m), 2.0, "labelled_target", paths[3]))
        for index, yaw_offset in enumerate((-30.0, 0.0, 30.0), start=1):
            yaw = (_yaw_to_target(*close_second, target.east_m, target.north_m) + yaw_offset) % 360.0
            waypoints.append(ObservationWaypoint(f"close_b_{index}", PHASES[2], *close_second, 1.5, yaw, 1.5, "labelled_target", paths[4] if index == 1 else ()))
        return tuple(waypoints)
    raise ValueError(f"no reachable observation route for {target.object_id}")


def _background_waypoints(manifest):
    points = ((3.5, 8.5), (3.5, 16.5), (3.5, 24.5), (3.5, 32.5))
    paths = _reachable_sequence(manifest, points)
    if paths is None:
        raise ValueError("background transit route is not reachable")
    phases = PHASES
    waypoints = []
    previous = (manifest.start_cell[0] + 0.5, manifest.start_cell[1] + 0.5)
    for index, (point, phase) in enumerate(zip(points, phases), start=1):
        east, north = point
        yaw = math.degrees(math.atan2(east - previous[0], north - previous[1])) % 360.0
        waypoints.append(ObservationWaypoint(f"background_{index}", phase, east, north, 1.5, yaw, 2.0, "verified_no_target", paths[index - 1]))
        previous = point
    return tuple(waypoints)


def build_visual_route(manifest, route_id, target_class):
    if not isinstance(manifest, LayoutManifest):
        raise TypeError("manifest must be a LayoutManifest")
    if target_class is None:
        target = None
        waypoints = _background_waypoints(manifest)
    else:
        targets = [item for item in manifest.objects if item.visual_category == target_class]
        if manifest.layout_seed % 2 == 0:
            targets.reverse()
        if not targets:
            raise ValueError(f"layout has no {target_class} target")
        target = None
        for candidate in targets:
            try:
                waypoints = _target_waypoints(manifest, candidate)
                target = candidate
                break
            except ValueError:
                continue
        if target is None:
            raise ValueError(f"layout has no reachable {target_class} target")
    return VisualRoute(
        route_id=str(route_id),
        layout_id=manifest.layout_id,
        target_class=target_class,
        target_object_id=None if target is None else target.object_id,
        start_cell=manifest.start_cell,
        waypoints=waypoints,
    )
