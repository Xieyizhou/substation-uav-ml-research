"""Lightweight check for the simulated perception state layer.

This script exercises the map-based detector and structured perception state
without PX4, Gazebo, or MAVSDK. It is useful for quickly verifying risk levels,
field-of-view filtering, and suggested actions after perception changes.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.perception_state import build_perception_state
from src.perception.simple_obstacle_detector import SimpleObstacleDetector
from src.planner.obstacle_config import build_obstacle_map, get_start_goal


class LocalPosition:
    """Minimal stand-in for MAVSDK local position telemetry."""

    def __init__(self, north_m, east_m, down_m=-1.5):
        self.north_m = north_m
        self.east_m = east_m
        self.down_m = down_m


class Attitude:
    """Minimal stand-in for MAVSDK Euler attitude telemetry."""

    def __init__(self, yaw_deg):
        self.yaw_deg = yaw_deg


def selected_obstacle_config_path():
    """Return the obstacle config paired with the currently selected Gazebo map."""
    from src.maps.map_catalog import current_map, project_path

    return project_path(current_map()["obstacle_config"])


def parse_args(argv=None):
    """Parse the standalone perception-smoke CLI."""
    parser = argparse.ArgumentParser(
        description="Run the perception smoke test against a selected or explicit map."
    )
    parser.add_argument(
        "--obstacle-config",
        type=Path,
        help=(
            "JSON obstacle config override. When omitted, use the config paired "
            "with the map selected by `python main.py map`."
        ),
    )
    return parser.parse_args(argv)


def resolve_config_path(config_path=None):
    """Resolve an optional CLI path, defaulting to the selected map config."""
    if config_path is None:
        return selected_obstacle_config_path()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path.resolve()


def load_config(config_path=None):
    """Load a selected or explicitly supplied obstacle configuration."""
    config_path = resolve_config_path(config_path)
    with config_path.open() as config_file:
        return json.load(config_file), config_path


def isolated_detector(obstacle_config):
    """Build a deterministic detector from one real cell in any catalog map.

    Isolating one physical cell keeps the five distance/FOV assertions stable
    even on dense maps where a neighboring obstacle would otherwise become the
    nearest detection first.
    """
    start, goal = get_start_goal(obstacle_config)
    obstacle_map = build_obstacle_map(
        obstacle_config,
        start_cell=start,
        goal_cell=goal,
    )
    raw_cells = sorted(obstacle_map["raw_obstacle_cells"])
    if not raw_cells:
        raise ValueError("Perception smoke test requires at least one raw obstacle cell")

    target_cell = raw_cells[0]
    target_name = obstacle_map["raw_obstacle_cell_to_name"].get(
        target_cell,
        "smoke_test_obstacle",
    )
    isolated_map = {
        "raw_obstacle_cells": {target_cell},
        "raw_obstacle_cell_to_name": {target_cell: target_name},
        "inflated_blocking_cells": set(),
        "inflated_obstacle_cell_to_name": {},
    }
    detector = SimpleObstacleDetector(
        obstacle_config,
        obstacle_map=isolated_map,
        detection_range_m=4.0,
        warning_distance_m=2.0,
        danger_distance_m=1.0,
        detection_fov_deg=90.0,
        use_raw_cells=True,
        use_inflated_cells=False,
    )
    return detector, detector.sensor_cells[0]


def state_for(detector, perception_config, north_m, east_m, yaw_deg):
    """Build a structured perception state for one synthetic pose."""
    position = LocalPosition(north_m=north_m, east_m=east_m)
    attitude = Attitude(yaw_deg=yaw_deg)
    detection = detector.detect(
        local_north_m=position.north_m,
        local_east_m=position.east_m,
        yaw_deg=attitude.yaw_deg,
        altitude_m=1.5,
    )
    return build_perception_state(
        perception_config,
        detection=detection,
        position=position,
        attitude=attitude,
    )


def assert_state(label, state, risk_level, in_range, in_fov, suggested_action):
    """Assert the expected risk classification for a named synthetic case."""
    assert state["enabled"] is True, label
    assert state["risk_level"] == risk_level, (label, state)
    assert state["in_detection_range"] is in_range, (label, state)
    assert state["in_fov"] is in_fov, (label, state)
    assert state["suggested_action"] == suggested_action, (label, state)
    assert "position" in state and "heading" in state, (label, state)


def run_smoke(obstacle_config, config_path=None):
    """Run clear, detected, warning, danger, and outside-FOV perception checks."""
    detector, target = isolated_detector(obstacle_config)
    perception_config = {
        "enabled": True,
        "risk_action": "slow_down",
    }

    north_m = target["obstacle_north_m"]
    east_m = target["obstacle_east_m"]
    clear = state_for(detector, perception_config, north_m, east_m - 5.0, 90.0)
    detected = state_for(detector, perception_config, north_m, east_m - 3.0, 90.0)
    warning = state_for(detector, perception_config, north_m, east_m - 1.5, 90.0)
    danger = state_for(detector, perception_config, north_m, east_m - 0.5, 90.0)
    outside_fov = state_for(detector, perception_config, north_m, east_m - 3.0, -90.0)

    assert_state("clear", clear, "clear", False, True, "none")
    assert_state("detected", detected, "detected", True, True, "none")
    assert_state("warning", warning, "warning", True, True, "slow_down")
    assert_state("danger", danger, "danger", True, True, "slow_down")
    assert_state("outside_fov", outside_fov, "clear", True, False, "none")

    if config_path is not None:
        map_name = obstacle_config.get("map_name", config_path.stem)
        print(f"Perception smoke map: {map_name}")
        print(f"Obstacle config: {config_path}")
    for label, state in [
        ("clear", clear),
        ("detected", detected),
        ("warning", warning),
        ("danger", danger),
        ("outside_fov", outside_fov),
    ]:
        print(
            f"{label}: risk={state['risk_level']} "
            f"value={state['risk_value']} "
            f"closest={state['closest_obstacle_name'] or 'none'} "
            f"distance={state['closest_obstacle_distance_m']} "
            f"in_range={state['in_detection_range']} "
            f"in_fov={state['in_fov']} "
            f"action={state['suggested_action']}"
        )


def main(argv=None):
    """Load the selected map and run the map-independent perception checks."""
    args = parse_args(argv)
    obstacle_config, config_path = load_config(args.obstacle_config)
    run_smoke(obstacle_config, config_path)


if __name__ == "__main__":
    main()
