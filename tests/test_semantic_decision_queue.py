import unittest

from src.flight.semantic_decision_queue import (
    decision_envelope,
    enqueue_decision,
    pending_queue_depth,
)


class SemanticDecisionQueueTests(unittest.TestCase):
    def test_identity_is_deterministic_and_duplicate_is_rejected(self):
        decision = {
            "kind": "exploration",
            "replacement_waypoints": [{"east_m": 1, "north_m": 2, "altitude_m": 2}],
        }
        first = decision_envelope(decision, created_at=1)
        second = decision_envelope(decision, created_at=9)
        self.assertEqual(first["decision_id"], second["decision_id"])
        self.assertEqual(first["route_identity"], second["route_identity"])
        config, state = {}, {}
        self.assertIsNotNone(enqueue_decision(config, state, decision, created_at=1))
        self.assertIsNone(enqueue_decision(config, state, decision, created_at=2))
        self.assertEqual(pending_queue_depth(config, state), 1)

    def test_consumed_index_controls_pending_depth(self):
        config = {"semantic_decisions": [{"decision_id": "one"}]}
        state = {"semantic_decision_index": 1}
        self.assertEqual(pending_queue_depth(config, state), 0)

    def test_queue_has_one_pending_slot_and_blocks_during_active_route(self):
        config, state = {}, {}
        first = {"decision_id": "one", "kind": "exploration", "replacement_waypoints": []}
        second = {"decision_id": "two", "kind": "exploration", "replacement_waypoints": []}
        self.assertIsNotNone(enqueue_decision(config, state, first, created_at=1))
        self.assertIsNone(enqueue_decision(config, state, second, created_at=2))
        state["semantic_decision_index"] = 1
        state["semantic_replacement_armed"] = False
        self.assertIsNone(enqueue_decision(config, state, second, created_at=3))


if __name__ == "__main__":
    unittest.main()
