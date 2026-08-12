import asyncio
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.flight.perception_response import current_perception_detection
from src.perception.lidar_detector import LidarRiskDetector
from src.perception.simple_obstacle_detector import SimpleObstacleDetector
from src.perception.local_costmap import RollingCostmapBuilder, build_local_costmap
from src.sensors.gazebo_lidar import parse_laser_scan_message, select_lidar_topic
from src.sensors.replay import ReplayLidarSource, append_scan_record, load_scan_records
from src.sensors.types import LaserScanFrame, SensorHealth


def scan_frame(ranges=(5.0, 1.0, 5.0), received=None):
    return LaserScanFrame(
        timestamp_s=100.0,
        received_monotonic_s=time.monotonic() if received is None else received,
        frame_id="lidar_link",
        angle_min_rad=-math.pi / 4,
        angle_max_rad=math.pi / 4,
        angle_step_rad=math.pi / 4,
        range_min_m=0.1,
        range_max_m=10.0,
        ranges_m=tuple(ranges),
        source="fixture",
        sequence=1,
    )


class FakeSource:
    source_id = "fixture_lidar"

    def __init__(self, frame, healthy=True):
        self.frame = frame
        self.healthy = healthy

    def latest(self):
        return self.frame

    def health(self, now_s=None):
        return SensorHealth(
            source=self.source_id,
            healthy=self.healthy,
            message="" if self.healthy else "stale",
            frequency_hz=20.0,
            last_frame_age_s=0.01 if self.healthy else 2.0,
        )


class SensorContractTests(unittest.TestCase):
    def test_map_oracle_accepts_the_shared_detector_velocity_keyword(self):
        detector = SimpleObstacleDetector.__new__(SimpleObstacleDetector)
        detector.warning_distance_m = 2.0
        detector.danger_distance_m = 1.0
        result = detector.detect(
            None, None, velocity_ned_m_s=(1.0, 0.0, 0.0)
        )
        self.assertEqual(result["risk_level"], "clear")

    def test_flight_perception_passes_ned_velocity_to_lidar_detector(self):
        detector = Mock()
        detector.detect.return_value = {
            "risk_level": "clear",
            "nearest_obstacle": None,
            "closest_obstacle": None,
        }
        velocity = SimpleNamespace(
            north_m_s=1.0, east_m_s=2.0, down_m_s=-0.5
        )
        current_perception_detection(
            {"enabled": True}, detector,
            SimpleNamespace(north_m=3.0, east_m=4.0, down_m=-1.5),
            SimpleNamespace(yaw_deg=15.0), velocity=velocity,
        )
        self.assertEqual(
            detector.detect.call_args.kwargs["velocity_ned_m_s"],
            (1.0, 2.0, -0.5),
        )

    def test_research_lidar_wins_over_generic_duplicate_topics(self):
        topics = [
            "/world/test/model/x500/link/link/sensor/lidar_2d_v2/scan",
            "/world/test/model/x500/link/research_lidar_link/sensor/lidar_2d_v2/scan",
            "/world/test/model/x500/link/lidar_sensor_link/sensor/lidar/scan",
        ]
        self.assertIn("research_lidar_link", select_lidar_topic(topics))

    def test_parse_gazebo_json_scan(self):
        frame = parse_laser_scan_message(
            {
                "angle_min": -1.0,
                "angle_max": 1.0,
                "angle_step": 1.0,
                "range_min": 0.1,
                "range_max": 12.0,
                "ranges": [2.0, 3.0, 4.0],
                "frame": "laser",
                "header": {"stamp": {"sec": 12, "nsec": 500_000_000}},
            },
            source="gazebo_lidar_2d",
            received_monotonic_s=7.0,
        )
        self.assertEqual(frame.ranges_m, (2.0, 3.0, 4.0))
        self.assertEqual(frame.frame_id, "laser")
        self.assertEqual(frame.timestamp_s, 12.5)

    def test_costmap_marks_hit_and_cleared_cells(self):
        costmap = build_local_costmap(
            scan_frame(),
            resolution_m=0.5,
            forward_range_m=6.0,
            lateral_range_m=6.0,
            inflation_radius_m=0.5,
        )
        self.assertEqual(len(costmap.occupancy), costmap.width * costmap.height)
        self.assertGreater(max(costmap.occupancy), 0.5)
        self.assertIn(False, costmap.unknown)

    def test_rolling_costmap_decays_unseen_obstacle_evidence(self):
        builder = RollingCostmapBuilder(
            resolution_m=0.5,
            forward_range_m=6.0,
            lateral_range_m=6.0,
            inflation_radius_m=0.0,
            decay_half_life_s=1.0,
        )
        first = builder.update(scan_frame((float("inf"), 2.0, float("inf")), received=1.0))
        second = builder.update(
            scan_frame((float("inf"), float("inf"), float("inf")), received=2.0)
        )
        occupied_index = first.occupancy.index(max(first.occupancy))
        self.assertGreater(first.occupancy[occupied_index], 0.5)
        self.assertGreater(second.occupancy[occupied_index], 0.5)
        self.assertLess(second.occupancy[occupied_index], first.occupancy[occupied_index])
        self.assertEqual(second.version, 2)

    def test_lidar_detector_produces_global_dynamic_cells(self):
        detector = LidarRiskDetector(
            FakeSource(scan_frame()),
            resolution_m=1.0,
            detection_range_m=6.0,
            warning_distance_m=2.0,
            danger_distance_m=0.5,
            nominal_speed_m_s=1.0,
        )
        detection = detector.detect(2.0, 3.0, yaw_deg=0.0)
        self.assertEqual(detection["risk_level"], "warning")
        self.assertTrue(detection["sensor_healthy"])
        self.assertTrue(detection["dynamic_grid_cells"])
        self.assertEqual(
            set(map(tuple, detection["dynamic_grid_cells"])),
            {
                (item["grid_x"], item["grid_y"])
                for item in detection["detected_obstacles"]
            },
        )
        self.assertIsNotNone(detection["costmap"])
        self.assertIn(detection["truth_risk_level"], {"clear", "warning", "danger"})
        self.assertEqual(detection["geometric_risk_level"], "warning")
        self.assertEqual(len(detection["truth_traversability"]), 72)
        self.assertEqual(
            detection["predicted_traversability"],
            detection["truth_traversability"],
        )

    def test_unhealthy_lidar_is_fail_safe_danger(self):
        detector = LidarRiskDetector(
            FakeSource(scan_frame(), healthy=False),
            resolution_m=1.0,
        )
        detection = detector.detect(0.0, 0.0, yaw_deg=0.0)
        self.assertEqual(detection["risk_level"], "danger")
        self.assertFalse(detection["sensor_healthy"])

    def test_record_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.jsonl"
            append_scan_record(path, scan_frame())
            frames = load_scan_records(path)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].ranges_m, (5.0, 1.0, 5.0))


class ReplaySourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_source_becomes_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.jsonl"
            append_scan_record(path, scan_frame())
            source = ReplayLidarSource(path, loop=True)
            await source.start()
            try:
                await source.wait_ready(0.5)
                self.assertTrue(source.health().healthy)
            finally:
                await source.stop()


if __name__ == "__main__":
    unittest.main()
