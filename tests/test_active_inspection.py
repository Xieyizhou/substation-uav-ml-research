import unittest

from src.flight.active_semantic_runtime import replacement_waypoints
from src.maps.map_draft import MapDraft, MapDraftCandidate
from src.planner.active_inspection import (
    ActiveInspectionPlanner,
    ActiveInspectionPolicy,
    RuntimeMap,
    run_active_inspection,
)


POLICY = {"runtime_mode": "active_semantic_inspection", "policy_id": "test", "model_artifact_identity": "a"*64,
          "weights": {"new_class": 4, "uninspected": 3, "confidence_gap": 2, "visibility_improvement": 1.5,
                      "path_length": -.08, "energy": -.12, "risk": -1, "repeat_observation": -.5}}
MAP = {"width_cells": 24, "height_cells": 24, "resolution_m": 1, "occupied_cells": [[12, 12]]}


def event(timestamp=1, class_name="transformer", tracking_id="t1", trigger="stable_detection"):
    return {"trigger": trigger, "timestamp_s": timestamp, "elapsed_s": timestamp,
            "vehicle_pose": {"east_m": 2, "north_m": 2, "altitude_m": 2, "yaw_deg": 0},
            "observations": [{"tracking_id": tracking_id, "class_name": class_name, "confidence": .8,
                              "bbox": [310, 200, 330, 240], "held": False, "stable": True, "depth_m": 7,
                              "camera_intrinsics": {"fx": 500, "cx": 320}}]}


class ActiveInspectionTests(unittest.TestCase):
    def test_frozen_weights(self):
        changed = {**POLICY, "weights": {**POLICY["weights"], "new_class": 5}}
        with self.assertRaises(ValueError):
            ActiveInspectionPolicy.from_mapping(changed)

    def test_determinism_and_merge(self):
        rows = [event(), event(2, tracking_id="t2")]
        first, second = run_active_inspection(MAP, POLICY, rows), run_active_inspection(MAP, POLICY, rows)
        self.assertEqual(first["artifact_identity"], second["artifact_identity"])
        self.assertEqual(first["tracks"][0]["observation_count"], 2)

    def test_class_conflict_and_held_isolation(self):
        result = run_active_inspection(MAP, POLICY, [event(), event(2, "reactor", "t1")])
        self.assertTrue(all(track["ambiguous"] for track in result["tracks"]))
        held = event(); held["observations"][0]["held"] = True
        self.assertEqual(run_active_inspection(MAP, POLICY, [held])["tracks"], [])

    def test_exploration_budget_and_safety_priority(self):
        empty = {**event(trigger="waypoint_reached"), "observations": []}
        budget = {**empty, "trigger": "budget_update", "timestamp_s": 600, "elapsed_s": 600}
        result = run_active_inspection(MAP, POLICY, [empty, budget])
        self.assertEqual(result["decisions"][0]["kind"], "exploration")
        self.assertEqual(result["events"][-1]["event"], "mission_budget_exhausted")
        safe = run_active_inspection(MAP, POLICY, [{**event(), "trigger": "safety_replan_active"}])
        self.assertFalse(any(row["event"] == "semantic_route_replaced" for row in safe["events"]))
        self.assertEqual(replacement_waypoints({}, safety_replan_active=True), [])

    def test_coverage_completion_waits_for_route_feedback(self):
        planner = ActiveInspectionPlanner(
            RuntimeMap.from_mapping(MAP), ActiveInspectionPolicy.from_mapping(POLICY)
        )
        planner.executed_coverage = .95
        stable = {**event(), "observations": []}
        planner.process(stable)
        self.assertFalse(planner.terminated)
        planner.process({**stable, "timestamp_s": 2, "elapsed_s": 2, "trigger": "waypoint_reached"})
        self.assertFalse(planner.terminated)
        self.assertEqual(planner.events[-2]["event"], "confirmation_sweep_started")
        planner.process({**stable, "timestamp_s": 3, "elapsed_s": 3, "trigger": "waypoint_reached"})
        self.assertTrue(planner.terminated)
        self.assertEqual(planner.events[-1]["event"], "exploration_completed")

    def test_map_draft_seam(self):
        candidate = MapDraftCandidate("c1", "transformer", .9, 1, 2, 3, 4, 5, 0, (0, 0, 20, 20))
        self.assertEqual(len(MapDraft("b"*64, 640, 480, (candidate,)).artifact_identity), 64)

    def test_active_route_registers_new_class_without_same_class_duplicate(self):
        planner = ActiveInspectionPlanner(
            RuntimeMap.from_mapping(MAP), ActiveInspectionPolicy.from_mapping(POLICY)
        )
        planner.route_active = True
        planner.process(event(tracking_id="replacement-id"))
        planner.process(event(2, tracking_id="replacement-id-2"))
        self.assertEqual(len(planner.tracks), 1)
        planner.process(event(3, class_name="reactor", tracking_id="reactor-id"))
        self.assertEqual(len(planner.tracks), 2)

    def test_inspected_track_absorbs_nearby_new_temporal_id(self):
        planner = ActiveInspectionPlanner(
            RuntimeMap.from_mapping(MAP), ActiveInspectionPolicy.from_mapping(POLICY)
        )
        first = event(tracking_id="first-id")
        planner.process(first)
        track = next(iter(planner.tracks.values()))
        track.inspected = True
        planner.route_active = False
        shifted = event(2, tracking_id="replacement-id")
        shifted["vehicle_pose"]["east_m"] += 6
        planner.process(shifted)
        self.assertEqual(len(planner.tracks), 1)

    def test_stable_target_id_completion_does_not_depend_on_temporal_id(self):
        planner = ActiveInspectionPlanner(
            RuntimeMap.from_mapping(MAP), ActiveInspectionPolicy.from_mapping(POLICY)
        )
        planner.process(event(tracking_id="expired-temporal-id"))
        target_id = next(iter(planner.tracks))
        planner.route_active = False
        feedback = {
            **event(2, trigger="target_completed"),
            "observations": [],
            "completed_target_ids": [target_id],
        }
        planner.process(feedback)
        self.assertTrue(planner.tracks[target_id].inspected)


if __name__ == "__main__":
    unittest.main()
