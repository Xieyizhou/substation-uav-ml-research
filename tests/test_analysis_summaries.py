import unittest

import pandas as pd

from src.logging.analysis_summaries import perception_summary, replan_summary


class AnalysisSummaryTests(unittest.TestCase):
    def test_perception_summary_counts_risk_and_distance_metrics(self):
        frame = pd.DataFrame(
            {
                "elapsed_s": [0.0, 1.0, 3.0],
                "perception_enabled": [True, True, True],
                "detected_obstacle": [False, True, True],
                "perception_risk_level": ["clear", "warning", "danger"],
                "nearest_obstacle_distance_m": [None, 1.2, 0.6],
                "nearest_obstacle_name": ["", "pole", "transformer"],
                "risk_action": ["slow_down"] * 3,
                "perception_source": ["gazebo_lidar_2d"] * 3,
                "sensor_healthy": [True, True, True],
                "sensor_frequency_hz": [28.0, 30.0, 32.0],
                "sensor_frame_age_s": [0.005, 0.010, 0.020],
                "sensor_dropped_frames": [0, 0, 1],
                "risk_model_id": ["geometric_lidar_v1"] * 3,
                "inference_latency_ms": [1.0, 2.0, 4.0],
                "costmap_version": [1, 2, 3],
                "equipment_detection_count": [0, 1, 1],
                "equipment_classes": ["", "transformer", "transformer|reactor"],
            }
        )
        summary = perception_summary(frame)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["samples_with_detections"], 2)
        self.assertEqual(summary["warning_sample_count"], 1)
        self.assertEqual(summary["danger_sample_count"], 1)
        self.assertAlmostEqual(summary["minimum_nearest_obstacle_distance_m"], 0.6)
        self.assertEqual(summary["nearest_obstacle_ever_detected"], "transformer")
        self.assertEqual(summary["perception_source"], "gazebo_lidar_2d")
        self.assertEqual(summary["sensor_healthy_ratio"], 1.0)
        self.assertAlmostEqual(summary["sensor_frame_age_p50_ms"], 10.0)
        self.assertEqual(summary["sensor_dropped_frames_max"], 1)
        self.assertEqual(summary["costmap_version_max"], 3)
        self.assertEqual(summary["equipment_classes"], ["reactor", "transformer"])

    def test_replan_summary_distinguishes_attempts_and_route_replacements(self):
        frame = pd.DataFrame(
            {
                "replan_triggered": [False, True, False],
                "replan_success": [False, True, False],
                "replan_count": [0, 1, 1],
                "replan_mode": ["active"] * 3,
                "replan_route_replaced": [False, True, False],
                "active_replan_count": [0, 1, 1],
                "active_replan_path_length": [None, 7, 7],
            }
        )
        summary = replan_summary(frame)
        self.assertEqual(summary["total_replan_attempts"], 1)
        self.assertEqual(summary["successful_replan_attempts"], 1)
        self.assertEqual(summary["active_route_replacement_count"], 1)
        self.assertEqual(summary["max_active_replan_path_length"], 7)


if __name__ == "__main__":
    unittest.main()
