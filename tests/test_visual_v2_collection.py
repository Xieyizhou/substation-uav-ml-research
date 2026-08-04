import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from src.cli.visual import build_parser
from src.flight.waypoint_executor import velocity_command_from_error
from src.ml import EQUIPMENT_CLASSES
from src.vision.collection.audit import audit_collection_plan
from src.vision.collection.layout import (
    LABEL_BY_CLASS,
    build_layout_manifest,
    validate_layout_manifest,
)
from src.vision.collection.plan import (
    build_collection_plan,
    load_collection_plan,
    validate_collection_plan,
    write_collection_plan,
)
from src.vision.collection.recording import (
    collection_recording_context,
    prepare_collection_scenario,
)
from src.vision.collection.route import build_visual_route
from src.vision.collection.timing import route_timing
from src.vision.collection.world import (
    build_layout_world,
    materialize_camera_model,
    validate_world_matches_layout,
)
from src.vision.contracts.protocol import load_protocol


class VisualV2CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol("v2")

    def layout(self, layout_id="development-01", split="development", seed=3001):
        return build_layout_manifest(
            layout_id,
            split,
            seed,
            self.protocol["randomization"]["configuration"],
        )

    def test_registered_v2_protocol_and_cli_selection(self):
        self.assertEqual(self.protocol.protocol_id, "visual-multiscenario-png-v2")
        self.assertEqual(self.protocol["split_policy"]["minimum_split_unit"], "layout")
        args = build_parser().parse_args(["collection-plan", "--protocol", "v2"])
        self.assertEqual(args.protocol, "v2")

    def test_v2_plan_is_deterministic_and_layout_isolation_is_strict(self):
        first = build_collection_plan("v2")
        second = build_collection_plan("v2")
        self.assertEqual(first, second)
        self.assertEqual(first["scenario_count"], 50)
        self.assertEqual(
            first["split_scenario_counts"],
            {"development": 30, "validation": 10, "blind": 10},
        )
        self.assertEqual(validate_collection_plan(first)["layout_count"], 10)
        layouts = {}
        for row in first["scenarios"]:
            layouts.setdefault(row["layout_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in layouts.values()))
        self.assertEqual({row["seed"] for row in first["scenarios"]}, set(range(3001, 3051)))

    def test_v1_plan_identity_remains_unchanged(self):
        plan = build_collection_plan("v1")
        self.assertEqual(
            plan["collection_plan_identity_sha256"],
            "b785c320ef83c92776ff84d601f3531173f641992a8d745cf66f5e082d5b2d66",
        )

    def test_plan_loader_resolves_protocol_from_plan(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            expected = write_collection_plan(path, "v2")
            self.assertEqual(load_collection_plan(path), expected)
            self.assertEqual(
                len(list((path.parent / "layouts").glob("*.json"))),
                10,
            )
            self.assertTrue((path.parent / "layout_index.json").is_file())
            layout_path = path.parent / "layouts/development-01.json"
            layout_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layout bundle mismatch"):
                load_collection_plan(path)

    def test_layout_inventory_labels_bounds_and_collisions(self):
        layout = self.layout()
        self.assertTrue(validate_layout_manifest(layout))
        counts = {name: 0 for name in EQUIPMENT_CLASSES}
        for item in layout.objects:
            if item.visual_category in counts:
                counts[item.visual_category] += 1
            self.assertEqual(item.simulator_label, LABEL_BY_CLASS.get(item.visual_category))
        self.assertTrue(all(count >= 2 for count in counts.values()))
        self.assertTrue({"cabinet", "pole", "control_building", "unknown_obstacle"}.issubset({item.visual_category for item in layout.objects}))

    def test_world_and_camera_noise_match_layout(self):
        layout = self.layout()
        world = build_layout_world(layout)
        self.assertTrue(validate_world_matches_layout(world, layout))
        source = Path("simulation/models/x500_research/model.sdf")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "model.sdf"
            materialize_camera_model(source, output, layout.camera_noise_stddev)
            stddev = ET.parse(output).findtext(".//sensor[@name='research_rgb']/camera/noise/stddev")
        self.assertAlmostEqual(float(stddev), layout.camera_noise_stddev)

    def test_target_routes_use_two_views_yaw_scan_and_stable_holds(self):
        layout = self.layout()
        for class_name in EQUIPMENT_CLASSES:
            route = build_visual_route(layout, f"{class_name}_centered_v2", class_name)
            close = [waypoint for waypoint in route.waypoints if waypoint.mission_phase == "close_inspection"]
            self.assertEqual(len(close), 6)
            self.assertTrue(all(waypoint.hold_s >= 1.5 for waypoint in close))
            first_yaws = [waypoint.yaw_deg for waypoint in close[:3]]
            second_yaws = [waypoint.yaw_deg for waypoint in close[3:]]
            self.assertAlmostEqual((first_yaws[1] - first_yaws[0]) % 360, 30.0)
            self.assertAlmostEqual((first_yaws[2] - first_yaws[1]) % 360, 30.0)
            view_delta = abs(((second_yaws[1] - first_yaws[1] + 180) % 360) - 180)
            self.assertAlmostEqual(view_delta, 90.0)

    def test_routes_keep_only_turning_cells_and_use_bounded_dynamic_timeout(self):
        layout = self.layout()
        route = build_visual_route(layout, "transformer_centered_v2", "transformer")
        self.assertLess(sum(len(item.transit_cells) for item in route.waypoints), 20)
        timing = route_timing(route, self.protocol["flight_timeout_policy"])
        self.assertGreaterEqual(timing["flight_timeout_s"], 180.0)
        self.assertLessEqual(timing["flight_timeout_s"], 360.0)
        self.assertLess(timing["flight_timeout_s"], 600.0)

    def test_background_route_is_no_target_and_faces_motion(self):
        layout = self.layout()
        route = build_visual_route(layout, "background_transit_v2", None)
        self.assertTrue(all(waypoint.expected_truth == "verified_no_target" for waypoint in route.waypoints))
        previous = (layout.start_cell[0] + 0.5, layout.start_cell[1] + 0.5)
        for waypoint in route.waypoints:
            expected = math.degrees(math.atan2(waypoint.east_m - previous[0], waypoint.north_m - previous[1])) % 360
            self.assertAlmostEqual(waypoint.yaw_deg, expected)
            previous = (waypoint.east_m, waypoint.north_m)

    def test_default_flight_yaw_remains_zero_and_explicit_yaw_is_used(self):
        error = {"north_m": 1.0, "east_m": 0.0, "down_m": 0.0}
        self.assertEqual(velocity_command_from_error(error).yaw_deg, 0.0)
        self.assertEqual(velocity_command_from_error(error, yaw_deg=123.0).yaw_deg, 123.0)

    def test_full_offline_collection_audit(self):
        result = audit_collection_plan(build_collection_plan("v2"))
        self.assertTrue(result["audit_passed"])
        self.assertEqual(result["layout_count"], 10)
        self.assertEqual(result["world_count"], 10)
        self.assertEqual(result["route_count"], 50)

    def test_v2_scenario_materializes_runtime_world_route_and_camera(self):
        plan = build_collection_plan("v2")
        row = plan["scenarios"][0]
        with TemporaryDirectory() as directory:
            prepared = prepare_collection_scenario(plan, row["scenario_id"], directory)
            for name in ("world_path", "planner_path", "route_path", "scenario_report"):
                self.assertTrue(Path(prepared[name]).is_file())
            self.assertTrue(Path(prepared["launcher_environment"]["RESEARCH_MODEL_SRC"]).is_file())
            self.assertIn("--visual-route", prepared["flight_command"])
            self.assertLess(prepared["flight_timeout_s"], 600.0)
            context = collection_recording_context(plan, row["scenario_id"], prepared["scenario_report"])
        self.assertEqual(context["protocol_id"], "visual-multiscenario-png-v2")
        self.assertEqual(context["map_id"], row["map_id"])
        self.assertEqual(context["target_id"], row["target_id"])


if __name__ == "__main__":
    unittest.main()
