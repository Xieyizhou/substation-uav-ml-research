from types import SimpleNamespace
import unittest

from src.flight.waypoint_executor import (
    takeoff_climb_waypoint,
    validate_takeoff_stability,
)


def _latest(*, north=0.0, east=0.0, down=-1.5, roll=0.0, pitch=0.0):
    return {
        "position_velocity": SimpleNamespace(
            position=SimpleNamespace(
                north_m=north,
                east_m=east,
                down_m=down,
            )
        ),
        "attitude": SimpleNamespace(roll_deg=roll, pitch_deg=pitch),
    }


class TakeoffStabilityTests(unittest.TestCase):
    def test_accepts_stable_takeoff(self):
        validate_takeoff_stability(_latest(), 1.5)
        validate_takeoff_stability(_latest(down=-0.1), 1.5)

    def test_rejects_excessive_attitude(self):
        with self.assertRaisesRegex(RuntimeError, "Takeoff is unstable"):
            validate_takeoff_stability(_latest(roll=45.0), 1.5)

    def test_rejects_horizontal_drift(self):
        with self.assertRaisesRegex(RuntimeError, "Takeoff drifted"):
            validate_takeoff_stability(_latest(north=3.0), 1.5)

    def test_rejects_implausible_altitude(self):
        with self.assertRaisesRegex(RuntimeError, "stability envelope"):
            validate_takeoff_stability(_latest(down=-4.0), 1.5)

    def test_takeoff_climb_holds_horizontal_position(self):
        waypoint = takeoff_climb_waypoint(
            _latest(north=0.2, east=-0.1),
            -1.5,
        )
        self.assertEqual(
            waypoint,
            {
                "name": "TAKEOFF_CLIMB",
                "north_m": 0.2,
                "east_m": -0.1,
                "down_m": -1.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
