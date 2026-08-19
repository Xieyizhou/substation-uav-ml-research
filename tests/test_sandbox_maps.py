import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET

from src.maps.sandbox_contracts import (
    SandboxMap, SandboxMapObject, SandboxMission,
)
from src.maps.sandbox_geometry import object_cells, objects_overlap
from src.maps.sandbox_materialize import materialize_revision
from src.maps.sandbox_flight import visual_route_for_sandbox
from src.maps.sandbox_routes import build_sandbox_route
from src.maps.sandbox_store import SandboxMapStore
from src.maps.sandbox_validation import validate_sandbox_map
from src.planner.obstacle_config import build_obstacle_map, load_obstacle_config


def sample_map(*, overlap=False, target_east=12.0):
    objects = [
        SandboxMapObject(
            "transformer_a", "transformer", target_east, 12.0,
            2.0, 2.0, 2.0, 30.0, "target",
        ),
        SandboxMapObject(
            "cabinet_a", "cabinet", 6.0 if not overlap else target_east,
            7.0 if not overlap else 12.0, 1.0, 1.0, 1.5,
        ),
    ]
    missions = (
        SandboxMission("go_north", "round_trip", 20.5, 20.5),
        SandboxMission(
            "inspect_transformer", "equipment_inspection",
            target_object_id="transformer_a",
        ),
    )
    return SandboxMap(
        "custom_station", "Custom station", 24.0, 24.0,
        1.5, 1.5, 0.0, tuple(objects), missions,
    )


class SandboxMapTests(unittest.TestCase):
    def test_identity_round_trip_is_stable(self):
        value = sample_map()
        restored = SandboxMap.from_record(value.to_record())
        self.assertEqual(restored, value)
        self.assertEqual(restored.map_identity_sha256, value.map_identity_sha256)

    def test_asset_label_roles_are_not_inferred(self):
        with self.assertRaisesRegex(ValueError, "label_role"):
            SandboxMapObject("transformer_a", "transformer", 5, 5, 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "label_role"):
            SandboxMapObject("cabinet_a", "cabinet", 5, 5, 1, 1, 1, label_role="target")

    def test_rotated_object_rasterization_is_conservative(self):
        item = sample_map().objects[0]
        cells = object_cells(item, 24, 24)
        self.assertGreaterEqual(len(cells), 4)
        self.assertIn((11, 11), cells)

    def test_overlap_and_unreachable_maps_are_rejected(self):
        value = sample_map(overlap=True)
        self.assertTrue(objects_overlap(value.objects[0], value.objects[1]))
        report, _ = validate_sandbox_map(value)
        self.assertFalse(report.valid)
        self.assertIn("object_overlap", {issue.code for issue in report.issues})

    def test_basic_and_inspection_routes_are_generated(self):
        value = sample_map()
        basic = build_sandbox_route(value, value.missions[0])
        inspection = build_sandbox_route(value, value.missions[1])
        self.assertTrue(basic.return_grid_path)
        self.assertEqual(inspection.mission_type, "equipment_inspection")
        self.assertEqual(len(inspection.waypoints), 5)
        self.assertTrue(inspection.return_grid_path)
        visual = visual_route_for_sandbox(value, value.missions[1], inspection)
        self.assertEqual(visual.target_class, "transformer")
        self.assertTrue(all(item.transit_cells for item in visual.waypoints))

    def test_revision_is_immutable_and_world_matches_planner(self):
        with TemporaryDirectory() as temporary:
            root, revision = materialize_revision(sample_map(), temporary)
            again, duplicate = materialize_revision(sample_map(), temporary)
            self.assertEqual(root, again)
            self.assertEqual(revision, duplicate)
            self.assertTrue((root / "preview.svg").is_file())
            world = ET.parse(root / "world.sdf")
            labels = world.findall(".//plugin[@name='gz::sim::systems::Label']/label")
            self.assertEqual([node.text for node in labels], ["1"])
            config = build_obstacle_map(load_obstacle_config(root / "obstacles.json"))
            obstacle_record = json.loads((root / "obstacles.json").read_text())
            expected = {
                tuple(cell)
                for item in obstacle_record["obstacles"]
                for cell in item["cells"]
            }
            self.assertEqual(config["raw_obstacle_cells"], expected)

    def test_store_round_trips_draft_revision_and_bundle(self):
        with TemporaryDirectory() as temporary, TemporaryDirectory() as imported:
            store = SandboxMapStore(temporary)
            value = store.save_draft(sample_map())
            self.assertEqual(store.read_draft(value.map_id), value)
            root, revision = store.create_revision(value.map_id)
            self.assertTrue((root / "identity.json").is_file())
            payload = store.export_bundle(value.map_id, revision.revision_identity_sha256)
            restored, identity = SandboxMapStore(imported).import_bundle(payload)
            self.assertEqual(restored, value)
            self.assertEqual(
                identity["revision_identity_sha256"],
                revision.revision_identity_sha256,
            )

    def test_store_rejects_path_traversal(self):
        with TemporaryDirectory() as temporary:
            store = SandboxMapStore(temporary)
            store.save_draft(sample_map())
            _, revision = store.create_revision("custom_station")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                store.revision_file(
                    "custom_station", revision.revision_identity_sha256,
                    "../../map.json",
                )


if __name__ == "__main__":
    unittest.main()
