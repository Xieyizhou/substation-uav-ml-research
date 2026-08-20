import tempfile
import unittest
from pathlib import Path

from src.vision.evaluation.temporal_observations import (
    TemporalObservationConfig,
    TemporalObservationFilter,
    evaluate_temporal_jsonl,
    evaluate_temporal_observations,
)


def prediction(confidence=0.8, x=0):
    return {
        "class_name": "transformer",
        "confidence": confidence,
        "bbox": [x, 0, x + 10, 10],
    }


class TemporalObservationTests(unittest.TestCase):
    def test_confirms_three_of_five_and_holds_for_200_ms(self):
        tracker = TemporalObservationFilter()
        self.assertEqual(tracker.update(0.0, [prediction()]), [])
        self.assertEqual(tracker.update(0.1, [prediction()]), [])
        confirmed = tracker.update(0.2, [prediction()])
        self.assertEqual(confirmed[0]["tracking_id"], "temporal-1")
        self.assertTrue(tracker.update(0.39, [], inferred=False)[0]["held"])
        self.assertEqual(tracker.update(0.401, [], inferred=False), [])

    def test_association_is_class_aware_and_confidence_is_smoothed(self):
        tracker = TemporalObservationFilter()
        for timestamp, confidence in ((0.0, 0.5), (0.1, 1.0), (0.2, 1.0)):
            rows = tracker.update(timestamp, [prediction(confidence)])
        self.assertAlmostEqual(rows[0]["confidence"], 0.92)
        other = {**prediction(), "class_name": "reactor"}
        tracker.update(0.3, [other])
        self.assertEqual(len(tracker._tracks), 2)

    def test_track_expires_and_is_not_reused(self):
        tracker = TemporalObservationFilter()
        for timestamp in (0.0, 0.1, 0.2):
            tracker.update(timestamp, [prediction()])
        tracker.update(0.71, [], inferred=False)
        self.assertEqual(tracker._tracks, [])
        tracker.update(0.8, [prediction()])
        self.assertEqual(tracker._tracks[0].track_id, 2)

    def test_reports_inference_and_timeline_recall_separately(self):
        truth = [{"class_name": "transformer", "bbox": [0, 0, 10, 10]}]
        frames = []
        for index in range(5):
            frames.append({
                "sample_id": f"f-{index}",
                "timestamp_s": index * 0.05,
                "inferred": index < 3,
                "truth": truth,
                "predictions": [prediction()] if index < 3 else [],
            })
        result = evaluate_temporal_observations(frames)
        self.assertEqual(result["inference_frame_recall"], 1.0)
        self.assertEqual(result["timeline_coverage_recall"], 0.6)
        self.assertEqual(result["observation_frame_count"], 3)

    def test_tracks_do_not_cross_recording_streams(self):
        frames = []
        for stream_id in ("recording-a", "recording-b"):
            for index in range(2):
                frames.append({
                    "sample_id": f"{stream_id}-{index}",
                    "stream_id": stream_id,
                    "timestamp_s": index * 0.1,
                    "predictions": [prediction()],
                })
        result = evaluate_temporal_observations(frames)
        self.assertEqual(result["observation_frame_count"], 0)

    def test_duplicate_sample_id_is_rejected(self):
        rows = [
            {"sample_id": "same", "timestamp_s": value, "predictions": []}
            for value in (0.0, 0.1)
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_temporal_observations(rows)

    def test_config_rejects_hold_beyond_expiration(self):
        with self.assertRaises(ValueError):
            TemporalObservationConfig(hold_ms=600, expiration_ms=500)

    def test_jsonl_materialization_binds_input_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.jsonl"
            source.write_text(
                '{"sample_id":"a","timestamp_s":0,"predictions":[]}\n'
            )
            result = evaluate_temporal_jsonl(source, root / "report.json")
            self.assertEqual(len(result["input_sha256"]), 64)
            self.assertTrue((root / "report.frames.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
