import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.flight.active_semantic_runtime import active_semantic_replacement
from src.planner.active_inspection import evaluate_plan, run_active_inspection
from src.study.active_inspection_benchmark import acceptance, compare_schedulers
from tests.test_active_inspection import MAP, POLICY, event


class ActiveInspectionIntegrationTests(unittest.TestCase):
    def test_four_classes_event_order_and_identity(self):
        rows = []
        classes = ("transformer", "reactor", "switchgear", "capacitor_bank")
        for index, class_name in enumerate(classes):
            row = event(index + 1, class_name, f"track-{index}")
            row["vehicle_pose"]["north_m"] = 2 + index * 4
            rows.append(row)
        first = run_active_inspection(MAP, POLICY, rows)
        second = run_active_inspection(MAP, POLICY, rows)
        self.assertEqual(first["artifact_identity"], second["artifact_identity"])
        self.assertEqual({track["class_name"] for track in first["tracks"]}, set(classes))
        event_names = [row["event"] for row in first["events"]]
        self.assertEqual(event_names.count("target_registered"), 4)
        self.assertLess(event_names.index("scheduler_decision"), event_names.index("semantic_route_replaced"))

    def test_offline_benchmark_has_all_three_schedulers(self):
        plan = run_active_inspection(MAP, POLICY, [event()])
        benchmark = compare_schedulers(plan, MAP)
        self.assertEqual(set(benchmark["baselines"]), {"fixed_serpentine", "nearest_target_first", "active_utility"})
        report = evaluate_plan(plan, MAP)
        result = acceptance(report, benchmark)
        self.assertEqual(result["evidence_level"], "deterministic_offline")
        self.assertFalse(result["passed"])


class ActiveSemanticFlightAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_replacement_uses_px4_format_and_is_consumed_once(self):
        drone = SimpleNamespace(offboard=SimpleNamespace(set_velocity_ned=AsyncMock()))
        collector = SimpleNamespace(events=[])
        collector.publish = lambda event_type, **details: collector.events.append({"event_type": event_type, **details})
        phase = {"_event_publisher": collector}
        config = {"semantic_runtime_mode": "active_semantic_inspection", "semantic_decisions": [{
            "decision_id": "d1", "candidate_id": "visual-target-1", "kind": "equipment_observation",
            "replacement_waypoints": [{"east_m": 3, "north_m": 4, "altitude_m": 2}],
        }]}
        state = {}
        replacement = await active_semantic_replacement(drone, phase, config, state, safety_replan_active=False)
        self.assertEqual(replacement[0]["down_m"], -2)
        self.assertEqual(replacement[0]["name"], "SIRWP01")
        self.assertIsNone(await active_semantic_replacement(drone, phase, config, state, safety_replan_active=False))
        drone.offboard.set_velocity_ned.assert_awaited_once()

    async def test_safety_blocks_semantic_queue_without_consuming_it(self):
        drone = SimpleNamespace(offboard=SimpleNamespace(set_velocity_ned=AsyncMock()))
        config = {"semantic_runtime_mode": "active_semantic_inspection", "semantic_decisions": [{
            "kind": "exploration", "replacement_waypoints": [{"east_m": 1, "north_m": 1, "altitude_m": 1}],
        }]}
        state = {}
        self.assertIsNone(await active_semantic_replacement(drone, {}, config, state, safety_replan_active=True))
        self.assertEqual(state.get("semantic_decision_index", 0), 0)
        drone.offboard.set_velocity_ned.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
