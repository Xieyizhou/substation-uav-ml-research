import json
import unittest

import pandas as pd

from src.logging.flight_logger import TELEMETRY_CSV_HEADER, perception_csv_values
from src.study.quality_metrics import lidar_quality_metrics


class StudyQualityMetricTests(unittest.TestCase):
    def test_quality_metrics_cover_risk_calibration_map_and_direction(self):
        clear = [1.0, 0.0]
        blocked = [0.0, 1.0]
        frame = pd.DataFrame([
            {
                "truth_risk_level": "clear",
                "perception_risk_level": "clear",
                "risk_confidence": 0.9,
                "truth_traversability_json": json.dumps(clear),
                "predicted_traversability_json": json.dumps(clear),
                "truth_direction_deg": 0.0,
                "predicted_direction_deg": 5.0,
            },
            {
                "truth_risk_level": "danger",
                "perception_risk_level": "clear",
                "risk_confidence": 0.8,
                "truth_traversability_json": json.dumps(blocked),
                "predicted_traversability_json": json.dumps(clear),
                "truth_direction_deg": 45.0,
                "predicted_direction_deg": 25.0,
            },
        ])
        metrics = lidar_quality_metrics(frame)
        self.assertEqual(metrics["quality_sample_count"], 2)
        self.assertEqual(metrics["danger_recall"], 0.0)
        self.assertEqual(metrics["truth_danger_sample_count"], 1)
        self.assertEqual(metrics["predicted_danger_sample_count"], 0)
        self.assertEqual(metrics["risk_false_negative_rate"], 1.0)
        self.assertEqual(metrics["traversability_iou"], 0.5)
        self.assertEqual(metrics["direction_mae_deg"], 12.5)
        self.assertIsNotNone(metrics["risk_ece"])

    def test_perception_csv_values_match_the_extended_schema(self):
        start = TELEMETRY_CSV_HEADER.index("perception_enabled")
        end = TELEMETRY_CSV_HEADER.index("replan_mode")
        values = perception_csv_values({"enabled": True}, None)
        self.assertEqual(len(values), end - start)

    def test_missing_truth_columns_produce_no_quality_claim(self):
        self.assertEqual(lidar_quality_metrics(pd.DataFrame([{"x": 1}])), {})


if __name__ == "__main__":
    unittest.main()
