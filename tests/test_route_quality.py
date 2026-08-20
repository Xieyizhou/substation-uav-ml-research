import json
from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from src.maps.sandbox_contracts import (
    SandboxMap, SandboxMapObject, SandboxMission,
)
from src.maps.sandbox_store import SandboxMapStore
from src.maps.route_quality import evaluate_route_quality
from src.maps.sandbox_routes import build_sandbox_route
from src.maps.sandbox_validation import validate_sandbox_map


def regression_map(index, category):
    width = 60.0 if category == "slow" else 20.0
    start = (2.5, 2.5)
    goal = (15.5, 15.5)
    speed = 1.0
    objects = ()
    if category == "normal":
        goals = ((15.5, 15.5), (16.5, 8.5), (8.5, 16.5))
        goal = goals[index % len(goals)]
    elif category == "same":
        goal = start
    elif category == "boundary":
        start, goal = (0.5, 0.5), (15.5, 0.5)
    elif category == "slow":
        goal, speed = (57.5, 57.5), 0.1
    elif category == "blocked":
        objects = (
            SandboxMapObject(
                "wall", "generic_obstacle", 10.0, 10.0,
                1.0, 20.0, 3.0,
            ),
        )
        goal = (17.5, 17.5)
    elif category == "narrow":
        start, goal = (2.5, 10.5), (17.5, 10.5)
        objects = (
            SandboxMapObject(
                "wall_low", "generic_obstacle", 10.0, 4.5,
                1.0, 9.0, 3.0,
            ),
            SandboxMapObject(
                "wall_high", "generic_obstacle", 10.0, 15.5,
                1.0, 9.0, 3.0,
            ),
        )
    mission = SandboxMission(
        "default_round_trip", "round_trip", *goal,
        speed_m_s=speed,
    )
    return SandboxMap(
        f"quality_{category}_{index:02d}", f"Quality {category} {index}",
        width, width, *start, 0.0, objects, (mission,),
    )


class RouteQualityRegressionTests(unittest.TestCase):
    def test_twenty_four_regression_maps_cover_quality_classes(self):
        cases = (
            [("normal", index, True, set()) for index in range(6)]
            + [("same", index, False, {"route_same_cell"}) for index in range(4)]
            + [("boundary", index, False, {"route_boundary_exposure_high"}) for index in range(4)]
            + [("slow", index, False, {"route_duration_excessive"}) for index in range(4)]
            + [("blocked", index, False, {"route_unreachable"}) for index in range(3)]
            + [("narrow", index, False, {"route_unreachable"}) for index in range(3)]
        )
        self.assertEqual(len(cases), 24)
        for category, index, expected_valid, required_codes in cases:
            with self.subTest(category=category, index=index):
                report, _ = validate_sandbox_map(regression_map(index, category))
                self.assertEqual(report.valid, expected_valid)
                self.assertTrue(required_codes <= {item.code for item in report.issues})

    def test_quality_artifact_is_identity_bound_and_required_for_flight(self):
        value = regression_map(0, "normal")
        with TemporaryDirectory() as temporary:
            store = SandboxMapStore(temporary)
            store.save_draft(value)
            root, revision = store.create_revision(value.map_id)
            quality_path = root / "routes/default_round_trip.quality.json"
            quality = json.loads(quality_path.read_text())
            self.assertTrue(quality["accepted"])
            self.assertEqual(
                revision.route_quality_identity_sha256,
                (quality["route_quality_identity_sha256"],),
            )
            _, _, route, accepted = store.accepted_route_artifacts(
                value.map_id, revision.revision_identity_sha256,
                "default_round_trip",
            )
            self.assertEqual(
                accepted["route_identity_sha256"], route["route_identity_sha256"]
            )

    def test_invalid_return_and_excessive_detour_are_rejected(self):
        value = regression_map(0, "normal")
        mission = value.missions[0]
        route = build_sandbox_route(value, mission)
        invalid_return = evaluate_route_quality(
            value, mission, replace(route, return_grid_path=())
        )
        self.assertIn(
            "return_route_invalid", {item.code for item in invalid_return.issues}
        )
        snake = []
        for north in range(2, 7):
            east_values = range(2, 18) if north % 2 == 0 else range(17, 1, -1)
            snake.extend((east, north) for east in east_values)
        snake.extend((17, north) for north in range(7, 16))
        snake.extend(((16, 15), (15, 15)))
        detour_route = replace(
            route,
            grid_path=tuple(snake),
            simplified_path=(snake[0], snake[-1]),
        )
        detour = evaluate_route_quality(value, mission, detour_route)
        self.assertGreater(detour.detour_ratio, 4.0)
        self.assertIn("route_detour_excessive", {item.code for item in detour.issues})

    def test_route_below_configured_clearance_is_rejected(self):
        obstacle = SandboxMapObject(
            "cabinet", "cabinet", 10.0, 10.0, 2.0, 2.0, 2.0,
        )
        mission = SandboxMission(
            "default_round_trip", "round_trip", 15.5, 15.5,
            horizontal_inflation_cells=1,
        )
        value = SandboxMap(
            "quality_clearance", "Quality clearance", 20.0, 20.0,
            2.5, 2.5, 0.0, (obstacle,), (mission,),
        )
        route = build_sandbox_route(value, mission)
        unsafe_path = (*route.grid_path[:2], (9, 9), *route.grid_path[2:])
        report = evaluate_route_quality(
            value, mission, replace(route, grid_path=unsafe_path)
        )
        self.assertEqual(report.minimum_clearance_m, 0.0)
        self.assertIn("route_clearance_low", {item.code for item in report.issues})

    def test_tampered_quality_artifact_is_rejected(self):
        value = regression_map(1, "normal")
        with TemporaryDirectory() as temporary:
            store = SandboxMapStore(temporary)
            store.save_draft(value)
            root, revision = store.create_revision(value.map_id)
            quality_path = root / "routes/default_round_trip.quality.json"
            record = json.loads(quality_path.read_text())
            record["accepted"] = False
            quality_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                store.accepted_route_artifacts(
                    value.map_id, revision.revision_identity_sha256,
                    "default_round_trip",
                )


if __name__ == "__main__":
    unittest.main()
