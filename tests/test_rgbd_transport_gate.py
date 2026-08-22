import unittest
from unittest.mock import AsyncMock, patch

from src.flight.active_semantic_runtime import complete_semantic_waypoint
from src.sensors.gazebo_visual_transport import discover_configured_topics
from src.sensors.rgb_depth_sync import RgbDepthSynchronizer


class TransportSnapshotTests(unittest.IsolatedAsyncioTestCase):
    @patch("src.sensors.gazebo_visual_transport._run_gz", new_callable=AsyncMock)
    async def test_single_snapshot_resolves_all_sources(self, run_gz):
        run_gz.return_value = "/rgb\n/depth\n/truth\n"
        result = await discover_configured_topics(
            {"rgb": "/rgb", "depth": "/depth", "truth": "/truth"}
        )
        self.assertEqual(result["resolved_topics"]["depth"], "/depth")
        run_gz.assert_awaited_once()

    @patch("src.sensors.gazebo_visual_transport._run_gz", new_callable=AsyncMock)
    async def test_ambiguous_snapshot_is_rejected(self, run_gz):
        run_gz.return_value = "/world/a/model/x/rgb\n/world/b/model/x/rgb\n"
        with self.assertRaises(RuntimeError):
            await discover_configured_topics({"rgb": "/model/x/rgb"})


class SynchronizerReceiptTests(unittest.TestCase):
    def test_receipt_reports_30_hz_pairing_and_skew(self):
        sync = RgbDepthSynchronizer(33.334)
        frame = lambda timestamp: type(
            "Frame", (), {"capture_timestamp": timestamp}
        )()
        for index in range(30):
            sync.push_depth(frame(index / 30 + .01))
            pairs = sync.push_rgb(frame(index / 30))
            self.assertEqual(len(pairs), 1)
        receipt = sync.receipt()
        self.assertEqual(receipt["rgb_pairing_success_rate"], 1)
        self.assertEqual(receipt["depth_pairing_success_rate"], 1)
        self.assertAlmostEqual(receipt["skew_ms"]["p95"], 10)


class SemanticFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_semantic_waypoint_publishes_completion(self):
        config = {"semantic_feedback_wait_s": 0}
        state = {
            "semantic_active_waypoint_count": 1,
            "semantic_active_decision": {
                "kind": "equipment_observation",
                "candidate_id": "target-1",
            },
        }
        result = await complete_semantic_waypoint(
            object(), object(), config, state, {"name": "SIRWP01"}, 4
        )
        self.assertIsNone(result)
        self.assertEqual(config["semantic_feedback"][0]["event"], "target_completed")
        self.assertEqual(config["semantic_feedback"][0]["candidate_id"], "target-1")


if __name__ == "__main__":
    unittest.main()
