#!/usr/bin/env python3
"""Generate a deterministic target-centered route for offline hard-example collection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planner.astar_grid import astar, simplify_grid_path
from src.planner.obstacle_config import build_obstacle_map
from src.vision.collection.route import ObservationWaypoint, VisualRoute


def _center(item):
    if item["type"] == "cell":
        return float(item["x"]) + .5, float(item["y"]) + .5
    return (float(item["x_min"]) + float(item["x_max"]) + 1) / 2, (float(item["y_min"]) + float(item["y_max"]) + 1) / 2


def _yaw(point, target):
    return math.degrees(math.atan2(target[0] - point[0], target[1] - point[1])) % 360


def _point(target, bearing, distance):
    angle = math.radians(bearing)
    return target[0] + math.sin(angle) * distance, target[1] + math.cos(angle) * distance


def generate(config_path, target_name, output, fixed_bearing=None, heading_offset=0.0):
    source = json.loads(config_path.read_text())
    target_row = next((row for row in source["obstacles"] if row["name"] == target_name), None)
    if target_row is None or target_row.get("type") != "rect" or not target_row.get("visual_category"):
        raise ValueError("target must be a named visual rectangle")
    target = _center(target_row); navigation = build_obstacle_map(source); blocked = navigation["inflated_blocking_cells"]
    start = tuple(source["start_cell"]); selected = None
    bearings = (float(fixed_bearing) % 360,) if fixed_bearing is not None else range(0, 360, 15)
    for bearing in bearings:
        points = (_point(target, bearing, 7), _point(target, bearing, 4.5), _point(target, bearing, 3), _point(target, (bearing + 90) % 360, 3))
        cells = [tuple(map(math.floor, point)) for point in points]
        if any(not (0 <= cell[0] < source["width"] and 0 <= cell[1] < source["height"]) or cell in blocked for cell in cells):
            continue
        current = start; paths = []
        try:
            for cell in cells:
                path = tuple(simplify_grid_path(astar(current, cell, blocked, source["width"], source["height"]))[1:])
                paths.append(path); current = cell
            return_path = tuple(simplify_grid_path(astar(current, start, blocked, source["width"], source["height"]))[1:])
        except ValueError:
            continue
        if return_path:
            selected = points, paths, return_path
            break
    if selected is None:
        raise ValueError("no reachable target-centered route")
    points, paths, return_path = selected
    specs = [
        ("distant", "cruise_distant", points[0], 15.0, 0.0),
        ("approach", "approach", points[1], 12.0, 0.0),
        ("close_a_left", "close_inspection", points[2], 12.0, -30.0),
        ("close_a_center", "close_inspection", points[2], 18.0, 0.0),
        ("close_a_right", "close_inspection", points[2], 12.0, 30.0),
        ("close_b_left", "target_transition", points[3], 12.0, -30.0),
        ("close_b_center", "close_inspection", points[3], 18.0, 0.0),
        ("close_b_right", "close_inspection", points[3], 12.0, 30.0),
    ]
    path_indexes = (0, 1, 2, None, None, 3, None, None)
    waypoints = []
    for spec, path_index in zip(specs, path_indexes):
        name, phase, point, hold, offset = spec
        labelled = target_row["visual_category"] in {
            "transformer", "switchgear", "capacitor_bank", "reactor"
        }
        expected_truth = (
            "labelled_target"
            if labelled and heading_offset % 360 == 0
            else "verified_no_target"
            if not labelled
            else "truth_audited"
        )
        waypoints.append(ObservationWaypoint(name, phase, point[0], point[1], 1.5, (_yaw(point, target) + heading_offset + offset) % 360, hold, expected_truth, () if path_index is None else paths[path_index]))
    bearing_suffix = "" if fixed_bearing is None else f"-bearing-{int(float(fixed_bearing) % 360):03d}"
    heading_suffix = "" if heading_offset % 360 == 0 else f"-heading-{int(float(heading_offset) % 360):03d}"
    route = VisualRoute(f"canonical-{target_name}{bearing_suffix}{heading_suffix}-hard-v1", f"canonical-{source['map_name']}", target_row["visual_category"], target_name, start, tuple(waypoints), return_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(route.to_record(), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"route":str(output),"identity":route.route_identity_sha256},sort_keys=True))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--target", required=True); parser.add_argument("--bearing", type=float); parser.add_argument("--heading-offset", type=float, default=0.0); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    generate(args.config, args.target, args.output, args.bearing, args.heading_offset); return 0


if __name__ == "__main__":
    raise SystemExit(main())
