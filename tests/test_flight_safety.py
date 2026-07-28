import unittest
from types import SimpleNamespace

from src.flight.flight_config import validate_planner_safety, validate_runtime_args


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
