import unittest
from types import SimpleNamespace

from src.flight.flight_config import (
    build_replan_config,
    validate_planner_safety,
    validate_runtime_args,
)


def valid_args(**overrides):
    values = {
        "max_speed": 0.8,
        "return_speed_scale": 0.7,
        "waypoint_acceptance": 0.3,
        "min_risk_speed": 0.3,
        "resolution": 1.0,
        "altitude": 1.5,
        "turn_settle": 1.0,
        "detection_range": 4.0,
        "detection_fov": 90.0,
        "warning_distance": 2.0,
        "danger_distance": 1.0,
        "connection_timeout": 30.0,
        "position_ready_timeout": 60.0,
        "telemetry_timeout": 10.0,
        "landing_timeout": 45.0,
        "logger_shutdown_timeout": 10.0,
        "enable_perception": True,
        "risk_action": "slow_down",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RuntimeArgumentValidationTests(unittest.TestCase):
    def test_valid_runtime_arguments_pass(self):
        validate_runtime_args(valid_args())

    def test_rejects_non_positive_values(self):
        for field in (
            "max_speed",
            "waypoint_acceptance",
            "resolution",
            "altitude",
            "connection_timeout",
            "telemetry_timeout",
            "landing_timeout",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_runtime_args(valid_args(**{field: 0}))

    def test_rejects_invalid_perception_distance_order(self):
        with self.assertRaisesRegex(ValueError, "danger <= warning <= detection"):
            validate_runtime_args(valid_args(danger_distance=3.0, warning_distance=2.0))

    def test_rejects_invalid_detection_fov(self):
        with self.assertRaisesRegex(ValueError, "detection-fov"):
            validate_runtime_args(valid_args(detection_fov=361.0))

    def test_rejects_minimum_risk_speed_above_maximum(self):
        with self.assertRaisesRegex(ValueError, "min-risk-speed"):
            validate_runtime_args(valid_args(min_risk_speed=0.9, max_speed=0.8))

    def test_dynamic_benchmark_requires_all_runtime_safety_controls(self):
        dynamic = {
            "dynamic_replan_scenario": "scenario.json",
            "enable_local_replan": True,
            "replan_mode": "active",
            "perception_source": "gazebo_lidar_2d",
            "return_home": True,
        }
        validate_runtime_args(valid_args(**dynamic))
        for field, value, message in (
            ("enable_local_replan", False, "active local replanning"),
            ("replan_mode", "log_only", "active local replanning"),
            ("perception_source", "map_baseline", "Gazebo LiDAR"),
            ("return_home", False, "return-home"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    invalid = {**dynamic, field: value}
                    validate_runtime_args(valid_args(**invalid))


class ReplanConfigurationTests(unittest.TestCase):
    def test_non_dynamic_configuration_cannot_replan_the_return_route(self):
        args = SimpleNamespace(
            replan_cooldown=1.0,
            dynamic_obstacle_inflation=1,
            max_replans=2,
            enable_local_replan=True,
            enable_perception=True,
            replan_mode="active",
            replan_risk_level="danger",
            allow_diagonal=True,
            dynamic_replan_scenario=None,
        )
        planner = {
            "width": 8,
            "height": 8,
            "resolution_m": 1.0,
            "altitude_m": 1.5,
            "goal": (6, 6),
            "start": (1, 1),
            "inflated_blocking_cells": {(3, 3)},
            "obstacle_map": object(),
        }
        config = build_replan_config(args, planner)
        self.assertFalse(config["allow_return_replan"])
        self.assertEqual(config["start_cell"], (1, 1))
        self.assertEqual(config["goal_cell"], (6, 6))


class PlannerSafetyValidationTests(unittest.TestCase):
    def planner(self, **overrides):
        values = {
            "start": (0, 0),
            "goal": (2, 2),
            "raw_obstacle_cells": {(1, 1)},
            "altitude_m": 1.5,
            "resolution_m": 1.0,
        }
        values.update(overrides)
        return values

    def test_safe_planner_configuration_passes(self):
        validate_planner_safety(self.planner())

    def test_start_inside_physical_obstacle_fails(self):
        with self.assertRaisesRegex(ValueError, "start cell"):
            validate_planner_safety(self.planner(raw_obstacle_cells={(0, 0)}))

    def test_goal_inside_physical_obstacle_fails(self):
        with self.assertRaisesRegex(ValueError, "goal cell"):
            validate_planner_safety(self.planner(raw_obstacle_cells={(2, 2)}))


if __name__ == "__main__":
    unittest.main()
