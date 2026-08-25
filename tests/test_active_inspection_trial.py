import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src.flight.active_inspection_trial import (
    load_active_inspection_trial,
    truncate_trial_route,
    wait_for_trial_budget,
)
from src.flight.active_semantic_runtime import (
    active_semantic_replacement,
    complete_semantic_waypoint,
)

TRIAL_PATH = Path("config/perception/active_inspection_trial_v1.json")


class ActiveInspectionTrialTests(unittest.IsolatedAsyncioTestCase):
    def test_multimap_qualification_is_frozen_with_yaw_scan(self):
        qualification = load_active_inspection_trial(
            "config/perception/active_inspection_multimap_qualification_policy_v1.json"
        )
        self.assertEqual(qualification["airborne_budget_s"], 600)
        self.assertEqual(qualification["max_semantic_replacements"], 10000)
        self.assertEqual(
            qualification["exploration_yaw_scan_deg"], [0, 90, 180, 270]
        )

    def test_formal_watchdog_is_frozen_without_route_caps(self):
        formal = load_active_inspection_trial(
            "config/perception/active_inspection_complex_formal_v1.json"
        )
        self.assertEqual(formal["airborne_budget_s"], 600)
        self.assertEqual(formal["max_semantic_replacements"], 10000)
        self.assertEqual(formal["max_route_distance_m"], 10000)

    def test_frozen_trial_and_route_truncation(self):
        trial = load_active_inspection_trial(TRIAL_PATH)
        route = [
            {"east_m": index, "north_m": 0, "altitude_m": 0}
            for index in range(12)
        ]
        result, truncated, distance = truncate_trial_route(route, 8, 1.5)
        self.assertTrue(truncated)
        self.assertEqual(distance, 8)
        self.assertEqual(len(result), 9)
        self.assertEqual({row["altitude_m"] for row in result}, {1.5})
        self.assertEqual(len(trial["artifact_identity"]), 64)

    async def test_budget_starts_only_after_takeoff(self):
        latest = {"in_air": False}
        task = asyncio.create_task(wait_for_trial_budget(latest, 0))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        latest["in_air"] = True
        await asyncio.wait_for(task, .2)

    @patch("src.flight.active_semantic_runtime.publish_mission_event")
    async def test_truncated_equipment_route_cannot_complete_target(self, publish):
        drone = SimpleNamespace(
            offboard=SimpleNamespace(set_velocity_ned=AsyncMock())
        )
        trial = load_active_inspection_trial(TRIAL_PATH)
        config = {
            "semantic_runtime_mode": "active_semantic_inspection",
            "semantic_trial": trial,
            "semantic_decisions": [{
                "decision_id": "d1",
                "candidate_id": "target-1",
                "kind": "equipment_observation",
                "replacement_waypoints": [
                    {"east_m": index, "north_m": 0, "altitude_m": 2}
                    for index in range(12)
                ],
            }],
        }
        state = {}
        replacement = await active_semantic_replacement(
            drone, {}, config, state, safety_replan_active=False
        )
        self.assertEqual(len(replacement), 9)
        state["semantic_active_waypoint_count"] = len(replacement)
        await complete_semantic_waypoint(
            drone, {}, config, state, replacement[-1], 10
        )
        self.assertEqual(config["semantic_feedback"][0]["event"], "waypoint_reached")
        self.assertFalse(any(
            call.args[1] == "target_completed" for call in publish.call_args_list
        ))

    async def test_confirmation_sweep_scans_each_marked_anchor(self):
        drone = SimpleNamespace(
            offboard=SimpleNamespace(set_velocity_ned=AsyncMock())
        )
        trial = load_active_inspection_trial(
            "config/perception/active_inspection_multimap_qualification_policy_v1.json"
        )
        config = {
            "semantic_runtime_mode": "active_semantic_inspection",
            "semantic_trial": trial,
            "semantic_decisions": [{
                "decision_id": "confirmation-1",
                "candidate_id": "confirmation:1",
                "kind": "exploration",
                "features": {"confirmation_sweep": True},
                "replacement_waypoints": [
                    {"east_m": 1, "north_m": 1, "altitude_m": 1.5, "confirmation_anchor": True},
                    {"east_m": 2, "north_m": 1, "altitude_m": 1.5},
                    {"east_m": 3, "north_m": 1, "altitude_m": 1.5, "confirmation_anchor": True},
                ],
            }],
        }
        replacement = await active_semantic_replacement(
            drone, {}, config, {}, safety_replan_active=False
        )
        self.assertEqual(len(replacement), 11)
        self.assertEqual(sum("dwell_s" in row for row in replacement), 8)


if __name__ == "__main__":
    unittest.main()
