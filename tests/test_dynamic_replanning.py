import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from src.flight.dynamic_blocker import (
    coordinate_dynamic_blocker,
    gazebo_spawn_command,
    load_dynamic_scenario,
    _run_command,
    _timeout_output,
    trigger_reached,
)
from src.flight.replanning_controller import route_allows_local_replan
from src.flight.active_replan_runtime import (
    active_replan_replacement,
    detection_with_verified_dynamic_occupancy,
)
from src.study.dynamic_replanning import (
    PHASE_EVENT_CHAIN,
    dynamic_benchmark_acceptance,
    dynamic_replanning_matrix,
    event_chain_report,
    materialize_dynamic_scenario,
)
from src.study.runner import _flight_arguments


class _EventCollector:
    def __init__(self):
        self.events = []

    def publish(self, event_type, **details):
        self.events.append({"event_type": event_type, **details})


class DynamicReplanningContractTests(unittest.TestCase):
    def test_detected_benchmark_blocker_uses_exact_simulator_cell(self):
        detection = {"risk_level": "warning", "dynamic_grid_cells": [[3, 4]]}
        config = {
            "dynamic_scenario": {"blocker": {"grid_cell": [7, 8]}},
        }

        enriched = detection_with_verified_dynamic_occupancy(detection, config)

        self.assertEqual(enriched["dynamic_grid_cells"], [(3, 4), (7, 8)])
        self.assertEqual(detection["dynamic_grid_cells"], [[3, 4]])

    def test_matrix_covers_three_maps_and_four_injection_phases(self):
        rows = dynamic_replanning_matrix()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["scenario_id"] for row in rows}), 12)
        self.assertEqual({row["map_id"] for row in rows}, {"simple", "medium", "complex"})
        self.assertEqual(
            {row["injection_phase"] for row in rows},
            {"early", "mid_route", "near_target", "return_leg"},
        )

    def test_materialization_is_stable_and_proves_an_alternative_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in dynamic_replanning_matrix():
                first = root / f"{row['scenario_id']}.first.json"
                second = root / f"{row['scenario_id']}.second.json"
                left = materialize_dynamic_scenario(
                    row["map_id"], row["injection_phase"], first
                )
                right = materialize_dynamic_scenario(
                    row["map_id"], row["injection_phase"], second
                )
                self.assertEqual(left, right)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertTrue(left["evidence"]["original_route_invalidated"])
                self.assertTrue(left["evidence"]["safe_alternative_exists"])

    def test_identity_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            materialize_dynamic_scenario("simple", "early", path)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["blocker"]["size_m"] = 9.0
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_dynamic_scenario(path)

    def test_gazebo_spawn_uses_the_bound_world_and_entity_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            scenario = materialize_dynamic_scenario("simple", "mid_route", path)
            command = gazebo_spawn_command(scenario)
            self.assertIn(f"/world/{scenario['world_name']}/create", command)
            self.assertIn("gz.msgs.EntityFactory", command)
            self.assertIn(scenario["blocker"]["id"], command[-1])

    @patch("src.flight.dynamic_blocker.subprocess.run")
    @patch("src.flight.dynamic_blocker.shutil.which", return_value="/opt/bin/gz")
    def test_gazebo_command_uses_spawn_safe_transport_environment(
        self, _which, run
    ):
        _run_command(["gz", "service", "-l"], 5.0)
        arguments, keywords = run.call_args
        self.assertEqual(arguments[0][0], "/opt/bin/gz")
        self.assertFalse(keywords["close_fds"])
        self.assertEqual(keywords["env"]["GZ_IP"], "127.0.0.1")

    def test_gazebo_timeout_output_accepts_bytes_and_text(self):
        error = TimeoutError()
        error.stdout = b"transport timed out"
        error.stderr = " while creating entity"
        self.assertEqual(
            _timeout_output(error),
            "transport timed out while creating entity",
        )

    def test_dynamic_flight_arguments_freeze_runtime_controls(self):
        arguments = _flight_arguments(
            "geometric_lidar", "model.onnx", "scenario.json",
            "planner.json", "dynamic.json",
        )
        for expected in (
            "--obstacle-config", "planner.json",
            "--dynamic-replan-scenario", "dynamic.json", "--return-home",
            "--max-replans", "1", "--detection-fov", "360",
            "--detection-range", "4", "--warning-distance", "2.5",
        ):
            self.assertIn(expected, arguments)

    def test_return_replanning_is_opt_in_for_dynamic_runs(self):
        self.assertFalse(route_allows_local_replan({"mode": "active"}, "return"))
        self.assertTrue(route_allows_local_replan(
            {"mode": "active", "allow_return_replan": True}, "return"
        ))

    def test_event_chain_requires_order_and_all_events(self):
        complete = [{"event_type": name} for name in PHASE_EVENT_CHAIN]
        self.assertTrue(event_chain_report(complete)["complete"])
        out_of_order = complete[1:3] + complete[:1] + complete[3:]
        report = event_chain_report(out_of_order)
        self.assertFalse(report["complete"])
        self.assertIn("dynamic_blocker_detected", report["missing_events"])

    def test_acceptance_rejects_missing_event_evidence(self):
        passing = [{
            "successful_replan": 1,
            "route_switch_correct": 1,
            "mission_success": 1,
            "landing_success": 1,
            "false_replan": 0,
            "collision_count": 0,
            "safety_failure_count": 0,
            "event_chain_complete": 1,
        } for _ in range(12)]
        self.assertTrue(dynamic_benchmark_acceptance(passing)["passed"])
        passing[0]["event_chain_complete"] = 0
        self.assertFalse(dynamic_benchmark_acceptance(passing)["passed"])

    def test_acceptance_reads_registry_result_envelopes(self):
        metrics = {
            "successful_replan": 1,
            "route_switch_correct": 1,
            "mission_success": 1,
            "landing_success": 1,
            "false_replan": 0,
            "collision_count": 0,
            "safety_failure_count": 0,
            "event_chain_complete": 1,
        }
        rows = [{"run_id": str(index), "metrics": metrics} for index in range(12)]
        self.assertTrue(dynamic_benchmark_acceptance(rows)["passed"])


class DynamicBlockerCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_dynamic_replan_waits_until_physical_blocker_is_spawned(self):
        drone = SimpleNamespace(offboard=SimpleNamespace(
            set_velocity_ned=AsyncMock()
        ))
        replacement = await active_replan_replacement(
            drone,
            {"route_direction": "outbound"},
            {
                "dynamic_scenario": {"blocker": {"id": "runtime_blocker"}},
                "mode": "active",
                "enabled": True,
            },
            {"replan_count": 0},
            None,
            {},
            "warning",
            "outbound",
            1.0,
        )
        self.assertIsNone(replacement)
        drone.offboard.set_velocity_ned.assert_not_awaited()

    async def test_trigger_spawns_once_and_publishes_provenance(self):
        scenario = {
            "route_direction": "outbound",
            "resolution_m": 1.0,
            "trigger": {"grid_cell": [3, 4], "radius_m": 0.5},
            "blocker": {"id": "runtime_blocker", "grid_cell": [6, 4]},
        }
        latest = {"position_velocity": SimpleNamespace(position=SimpleNamespace(
            east_m=3.5, north_m=4.5
        ))}
        collector = _EventCollector()
        phase_state = {
            "phase": "outbound_to_goal",
            "route_direction": "outbound",
            "_event_publisher": collector,
        }
        spawn = AsyncMock(return_value="data: true")
        self.assertTrue(trigger_reached(latest, phase_state, scenario))
        await coordinate_dynamic_blocker(
            latest, phase_state, scenario, spawn=spawn, poll_s=0
        )
        spawn.assert_awaited_once_with(scenario)
        self.assertTrue(phase_state["_dynamic_blocker_spawned"])
        self.assertEqual(collector.events[0]["event_type"], "dynamic_blocker_spawned")

    async def test_trigger_rejects_the_wrong_route_direction(self):
        scenario = {
            "route_direction": "return",
            "resolution_m": 1.0,
            "trigger": {"grid_cell": [0, 0], "radius_m": 2.0},
        }
        latest = {"position_velocity": SimpleNamespace(position=SimpleNamespace(
            east_m=0.5, north_m=0.5
        ))}
        self.assertFalse(trigger_reached(
            latest, {"route_direction": "outbound"}, scenario
        ))


if __name__ == "__main__":
    unittest.main()
