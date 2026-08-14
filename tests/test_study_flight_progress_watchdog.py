from pathlib import Path
import tempfile
import unittest

from src.study.flight_progress_watchdog import (
    FlightProgress,
    FlightProgressReader,
    FlightProgressWatchdog,
)


class FlightProgressWatchdogTests(unittest.TestCase):
    def test_extends_soft_timeout_while_log_and_route_progress_continue(self):
        watchdog = FlightProgressWatchdog(0.0, 100.0)
        watchdog.observe_activity(99.0)
        watchdog.observe_progress(
            FlightProgress(("return", "return", "WP02"), 6.0), 99.0
        )
        watchdog.observe_activity(110.0)
        watchdog.observe_progress(
            FlightProgress(("return", "return", "WP02"), 5.5), 110.0
        )
        self.assertTrue(watchdog.should_continue(111.0))

    def test_stops_extension_when_progress_stalls(self):
        watchdog = FlightProgressWatchdog(0.0, 100.0)
        watchdog.observe_activity(140.0)
        watchdog.observe_progress(
            FlightProgress(("return", "return", "WP02"), 6.0), 90.0
        )
        self.assertFalse(watchdog.should_continue(140.0))
        self.assertIn("no 0.25 m", watchdog.stop_reason(140.0))

    def test_absolute_limit_stops_even_with_recent_progress(self):
        watchdog = FlightProgressWatchdog(0.0, 100.0)
        watchdog.observe_activity(199.0)
        watchdog.observe_progress(
            FlightProgress(("return", "return", "WP02"), 5.0), 199.0
        )
        self.assertFalse(watchdog.should_continue(200.0))
        self.assertEqual(watchdog.stop_reason(200.0), "absolute flight limit reached")

    def test_reader_tracks_appended_complete_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flight.csv"
            path.write_text(
                "phase,route_direction,target_name,horizontal_error_m\n"
                "return,return,WP02,6.0\n",
                encoding="utf-8",
            )
            reader = FlightProgressReader(path)
            self.assertEqual(reader.read_latest().horizontal_error_m, 6.0)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("return,return,WP02,5.5\n")
            self.assertEqual(reader.read_latest().horizontal_error_m, 5.5)
            reader.close()


if __name__ == "__main__":
    unittest.main()
