from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import os
import unittest
from unittest.mock import patch

from src.flight.flight_config import make_log_path
from src.flight.live_telemetry import snapshot_record, write_snapshot


class LiveTelemetryTests(unittest.TestCase):
    def test_snapshot_is_atomic_and_contains_flight_pose(self):
        start = datetime.now(timezone.utc)
        now = start + timedelta(seconds=1.25)
        record = snapshot_record(
            now, start, {"phase": "outbound", "route_direction": "forward"},
            {"name": "waypoint", "north_m": 3, "east_m": 4, "down_m": -1.5},
            SimpleNamespace(north_m=1, east_m=2, down_m=-1.5),
            SimpleNamespace(north_m_s=0.4, east_m_s=0.3, down_m_s=0),
            SimpleNamespace(yaw_deg=45),
            {"connected": True, "armed": True, "in_air": True},
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "live.json"
            write_snapshot(path, record)
            restored = json.loads(path.read_text())
            self.assertEqual(restored["elapsed_s"], 1.25)
            self.assertEqual(restored["position"]["east_m"], 2.0)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_map_run_can_bind_telemetry_to_its_output_directory(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "run" / "telemetry.csv"
            with patch.dict(os.environ, {"UAV_TELEMETRY_LOG_PATH": str(target)}):
                self.assertEqual(make_log_path(), target)


if __name__ == "__main__":
    unittest.main()
