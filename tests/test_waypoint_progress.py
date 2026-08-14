import unittest

from src.flight.waypoint_progress import (
    WaypointProgressWatchdog,
    parse_waypoint_timeout,
)


class WaypointProgressWatchdogTests(unittest.TestCase):
    def test_auto_timeout_continues_while_distance_improves(self):
        watchdog = WaypointProgressWatchdog(
            0.0, 20.0, 12.0, enabled=True,
        )
        for now_s, distance_m in ((10.0, 10.0), (19.0, 8.0), (30.0, 6.0)):
            watchdog.observe(distance_m, now_s)
        self.assertTrue(watchdog.should_continue(31.0))

    def test_auto_timeout_stops_after_progress_stalls(self):
        watchdog = WaypointProgressWatchdog(
            0.0, 20.0, 12.0, enabled=True,
        )
        watchdog.observe(10.0, 10.0)
        self.assertFalse(watchdog.should_continue(40.1))
        self.assertIn("no 0.25 m progress", watchdog.stop_reason(40.1))

    def test_absolute_limit_stops_continuous_progress(self):
        watchdog = WaypointProgressWatchdog(
            0.0, 100.0, 12.0, enabled=True,
        )
        watchdog.observe(11.0, 299.0)
        self.assertFalse(watchdog.should_continue(300.0))
        self.assertEqual(
            watchdog.stop_reason(300.0), "absolute waypoint limit reached"
        )

    def test_explicit_timeout_remains_strict(self):
        watchdog = WaypointProgressWatchdog(
            0.0, 20.0, 12.0, enabled=False,
        )
        watchdog.observe(5.0, 19.0)
        self.assertFalse(watchdog.should_continue(20.0))

    def test_timeout_parser_preserves_auto_and_positive_values(self):
        self.assertEqual(parse_waypoint_timeout("auto"), "auto")
        self.assertEqual(parse_waypoint_timeout("12.5"), 12.5)
        with self.assertRaises(ValueError):
            parse_waypoint_timeout("0")


if __name__ == "__main__":
    unittest.main()
