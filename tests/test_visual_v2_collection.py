import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from src.cli.visual import build_parser
from src.flight.waypoint_executor import normalize_yaw_deg, velocity_command_from_error
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
from src.vision.collection.v2_recording import _spawn_pose
from src.vision.collection.route import (
    ObservationWaypoint,
    VisualRoute,
    _yaw_to_target,
    build_visual_route,
)
from src.vision.collection.timing import route_timing
from src.vision.collection.flight_route import (
    _cell_waypoint,
    _observation_waypoint,
)
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
        self.assertEqual(
            self.protocol["recording"]["camera_heading_offset_deg"],
            341.0,
        )
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
        for item in layout.objects:
            model = world.find(
                f"./world/model[@name='{item.object_id}']"
            )
            self.assertIsNotNone(model)
            self.assertIsNone(model.find("./plugin"))
            east, north = (float(value) for value in model.findtext("./pose").split()[:2])
            self.assertAlmostEqual(east, item.east_m - layout.width_m / 2, places=4)
            self.assertAlmostEqual(north, item.north_m - layout.height_m / 2, places=4)
            label = model.findtext("./link/visual/plugin/label")
            expected = LABEL_BY_CLASS.get(item.visual_category)
            self.assertEqual(label, None if expected is None else str(expected))
        source = Path("simulation/models/x500_research/model.sdf")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "model.sdf"
            materialize_camera_model(
                source,
                output,
                layout.camera_noise_stddev,
                pitch_down_deg=5.0,
            )
            tree = ET.parse(output)
            stddev = tree.findtext(
                ".//sensor[@name='research_rgb']/camera/noise/stddev"
            )
            pose = tree.findtext(".//link[@name='research_camera_link']/pose")
            cameras = {
                sensor.get("name"): sensor.find("camera")
                for sensor in tree.findall(".//sensor")
                if sensor.get("name") in {"research_rgb", "research_boxes"}
            }
            sensor_masses = {
                link.get("name"): float(link.findtext("inertial/mass"))
                for link in tree.findall(".//link")
                if link.get("name") in {
                    "research_camera_link",
                    "research_lidar_link",
                }
            }
        self.assertAlmostEqual(float(stddev), layout.camera_noise_stddev)
        self.assertAlmostEqual(float(pose.split()[4]), math.radians(5.0))
        self.assertEqual(
            sensor_masses,
            {
                "research_camera_link": 0.001,
                "research_lidar_link": 0.001,
            },
        )
        for camera in cameras.values():
            self.assertAlmostEqual(
                float(camera.findtext("lens/intrinsics/cx")),
                960.0,
            )
            self.assertAlmostEqual(
                float(camera.findtext("lens/intrinsics/cy")),
                540.0,
            )
            self.assertAlmostEqual(
                float(camera.findtext("lens/intrinsics/fx")),
                float(camera.findtext("lens/intrinsics/fy")),
            )
        first_model = world.find(
            f"./world/model[@name='{layout.objects[0].object_id}']"
        )
        first_model.find("./pose").text = "0 0 0 0 0 0"
        with self.assertRaisesRegex(ValueError, "pose mismatch"):
            validate_world_matches_layout(world, layout)

    def test_target_routes_use_two_views_yaw_scan_and_stable_holds(self):
        layout = self.layout()
        for class_name in EQUIPMENT_CLASSES:
            route = build_visual_route(layout, f"{class_name}_centered_v2", class_name)
            close = [waypoint for waypoint in route.waypoints if waypoint.mission_phase == "close_inspection"]
            self.assertEqual(len(close), 6)
            self.assertTrue(all(waypoint.hold_s >= 2.5 for waypoint in close))
            self.assertEqual(route.waypoints[0].waypoint_id, "small_scale")
            self.assertEqual(route.waypoints[0].hold_s, 6.0)
            target = next(
                item
                for item in layout.objects
                if item.object_id == route.target_object_id
            )
            small_scale_distance = math.hypot(
                route.waypoints[0].east_m - target.east_m,
                route.waypoints[0].north_m - target.north_m,
            )
            self.assertAlmostEqual(small_scale_distance, 18.0)
            self.assertEqual(route.waypoints[1].hold_s, 10.0)
            self.assertEqual(route.waypoints[2].hold_s, 5.0)
            first_yaws = [waypoint.yaw_deg for waypoint in close[:3]]
            second_yaws = [waypoint.yaw_deg for waypoint in close[3:]]
            self.assertAlmostEqual((first_yaws[1] - first_yaws[0]) % 360, 30.0)
            self.assertAlmostEqual((first_yaws[2] - first_yaws[1]) % 360, 30.0)
            view_delta = abs(((second_yaws[1] - first_yaws[1] + 180) % 360) - 180)
            self.assertAlmostEqual(view_delta, 90.0)

    def test_layout_axes_match_px4_local_ned_at_flight_boundary(self):
        cell = _cell_waypoint((7, 11), 1.5, 30.0, "cell", (0.0, 0.0))
        self.assertEqual((cell["north_m"], cell["east_m"]), (11.5, 7.5))
        observation = ObservationWaypoint(
            "point", "approach", 8.0, 12.0, 1.5, 45.0, 2.0,
            "labelled_target", (),
        )
        mapped = _observation_waypoint(observation, (0.0, 0.0))
        self.assertEqual((mapped["north_m"], mapped["east_m"]), (12.0, 8.0))

        origin = (2.5, 2.5)
        local_cell = _cell_waypoint((7, 11), 1.5, 30.0, "cell", origin)
        self.assertEqual(
            (local_cell["north_m"], local_cell["east_m"]),
            (9.0, 5.0),
        )
        local_observation = _observation_waypoint(observation, origin)
        self.assertEqual(
            (local_observation["north_m"], local_observation["east_m"]),
            (9.5, 5.5),
        )

    def test_v2_spawn_pose_is_center_of_planner_start_cell(self):
        layout = self.layout()
        self.assertEqual(_spawn_pose(layout, 0.1), "-17.5,-17.5,0.1,0,0,0")

        route = build_visual_route(
            layout,
            "transformer_centered_v2",
            "transformer",
        )
        origin = (layout.start_cell[0] + 0.5, layout.start_cell[1] + 0.5)
        local = _observation_waypoint(route.waypoints[0], origin)
        spawn_east, spawn_north = (-17.5, -17.5)
        self.assertAlmostEqual(
            spawn_east + local["east_m"],
            route.waypoints[0].east_m - layout.width_m / 2,
        )
        self.assertAlmostEqual(
            spawn_north + local["north_m"],
            route.waypoints[0].north_m - layout.height_m / 2,
        )

    def test_target_yaw_uses_px4_ned_axes(self):
        self.assertEqual(_yaw_to_target(0, 1, 0, 0), 180.0)
        self.assertEqual(_yaw_to_target(0, -1, 0, 0), 0.0)
        self.assertEqual(_yaw_to_target(0, 0, 1, 0), 90.0)
        self.assertEqual(_yaw_to_target(0, 0, -1, 0), 270.0)
        layout = self.layout()
        route = build_visual_route(layout, "transformer_centered_v2", "transformer")
        target = next(item for item in layout.objects if item.object_id == route.target_object_id)
        distant = route.waypoints[0]
        expected = math.degrees(math.atan2(
            target.east_m - distant.east_m,
            target.north_m - distant.north_m,
        )) % 360.0
        self.assertAlmostEqual(distant.yaw_deg, expected)
        camera_route = build_visual_route(
            layout,
            "transformer_centered_v2",
            "transformer",
            camera_heading_offset_deg=180.0,
        )
        self.assertAlmostEqual(
            (camera_route.waypoints[0].yaw_deg - expected) % 360.0,
            180.0,
        )

    def test_routes_keep_only_turning_cells_and_use_bounded_dynamic_timeout(self):
        layout = self.layout()
        route = build_visual_route(layout, "transformer_centered_v2", "transformer")
        self.assertLess(sum(len(item.transit_cells) for item in route.waypoints), 20)
        timing = route_timing(route, self.protocol["flight_timeout_policy"])
        self.assertGreaterEqual(timing["flight_timeout_s"], 180.0)
        self.assertLessEqual(timing["flight_timeout_s"], 360.0)
        self.assertLess(timing["flight_timeout_s"], 600.0)
        with self.assertRaisesRegex(ValueError, "yaw tolerance"):
            VisualRoute(
                "invalid",
                layout.layout_id,
                None,
                None,
                layout.start_cell,
                route.waypoints,
                route.return_transit_cells,
                yaw_tolerance_deg=0.0,
            )
        with self.assertRaisesRegex(ValueError, "acquisition timeout"):
            VisualRoute(
                "invalid",
                layout.layout_id,
                None,
                None,
                layout.start_cell,
                route.waypoints,
                route.return_transit_cells,
                yaw_settle_duration_s=2.0,
                yaw_acquisition_timeout_s=1.0,
            )

    def test_background_route_is_no_target_and_faces_motion(self):
        layout = self.layout()
        route = build_visual_route(layout, "background_transit_v2", None)
        self.assertTrue(all(waypoint.expected_truth == "verified_no_target" for waypoint in route.waypoints))
        previous = (layout.start_cell[0] + 0.5, layout.start_cell[1] + 0.5)
        for waypoint in route.waypoints:
            expected = math.degrees(math.atan2(
                waypoint.east_m - previous[0],
                waypoint.north_m - previous[1],
            )) % 360
            self.assertAlmostEqual(waypoint.yaw_deg, expected)
            previous = (waypoint.east_m, waypoint.north_m)

    def test_default_flight_yaw_remains_zero_and_explicit_yaw_is_used(self):
        error = {"north_m": 1.0, "east_m": 0.0, "down_m": 0.0}
        self.assertEqual(velocity_command_from_error(error).yaw_deg, 0.0)
        self.assertEqual(velocity_command_from_error(error, yaw_deg=123.0).yaw_deg, 123.0)
        self.assertEqual(normalize_yaw_deg(225.0), -135.0)
        self.assertEqual(normalize_yaw_deg(180.0), 180.0)

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
            self.assertEqual(
                prepared["launcher_environment"]["PX4_GZ_MODEL_POSE"],
                "-17.5,-17.5,0.1,0,0,0",
            )
            self.assertIn("--visual-route", prepared["flight_command"])
            self.assertLess(prepared["flight_timeout_s"], 600.0)
            context = collection_recording_context(plan, row["scenario_id"], prepared["scenario_report"])
        self.assertEqual(context["protocol_id"], "visual-multiscenario-png-v2")
        self.assertEqual(context["map_id"], row["map_id"])
        self.assertEqual(context["target_id"], row["target_id"])

if __name__ == "__main__":
    unittest.main()
