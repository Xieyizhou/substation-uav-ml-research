import json
from pathlib import Path
import tempfile
import unittest

from src.vision.canonical.gates import (
    CHECK_VERSION, annotation_mode_from_world, instance_mapping, target_checks,
    validate_point, validate_preflight,
)
from src.vision.canonical.plan import write_record
from src.vision.canonical.recovery import resumed_views
from src.ml.artifacts import file_sha256


WORLD = """<sdf><world name="test"><model name="camera"><link name="link">
<sensor name="boxes" type="boundingbox_camera"><camera><box_type>{mode}</box_type></camera></sensor>
</link></model></world></sdf>"""


def config():
    return {"gazebo_world_origin_m": [0, 0, 0], "width": 10, "height": 10,
            "obstacles": [{"name": "box", "x": 4, "y": 4, "z_max_m": 2}]}


def obj(name="transformer_a", label="51"):
    return {"name": name, "category": "transformer", "runtime_labels": [label]}


def view():
    return {"view_id": "v", "object_id": "transformer_a", "category": "transformer",
            "position": [2, 2, 2], "camera_position": [2.1, 2, 2]}


class CanonicalGateTests(unittest.TestCase):
    def test_world_mode_is_read_and_declaration_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "world.sdf").write_text(WORLD.format(mode="visible_2d"))
            (root / "obstacles.json").write_text(json.dumps(config()))
            plan = {"annotation_mode": "full_2d", "label_mode": "visual-instance",
                    "hierarchy_mode": "top-level-equipment", "objects": [obj()]}
            self.assertEqual(annotation_mode_from_world(root / "world.sdf"), "visible_2d")
            with self.assertRaisesRegex(ValueError, "Declared/world"):
                validate_preflight(plan, root, [view()])

    def test_missing_duplicate_unknown_camera_modes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.sdf"
            for payload in ("<sdf><world/></sdf>",
                            WORLD.format(mode="mystery"),
                            "<sdf><world>" + WORLD.format(mode="full_2d").split("<world name=\"test\">")[1].split("</world>")[0] * 2 + "</world></sdf>"):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    annotation_mode_from_world(path)

    def test_mapping_collision_and_unresolvable_fail(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            instance_mapping({"objects": [obj("a"), obj("b")]})
        with self.assertRaisesRegex(ValueError, "Unresolvable"):
            instance_mapping({"objects": [obj(label=None)]})

    def test_same_class_cannot_replace_planned_instance(self):
        mapping = {51: {"object_id": "transformer_a", "category": "transformer"},
                   52: {"object_id": "transformer_b", "category": "transformer"}}
        checks = target_checks(view(), {"annotated_box": [{"label": 52}]}, mapping)
        self.assertTrue(checks["category_present"])
        self.assertFalse(checks["planned_instance_present"])
        self.assertEqual(checks["visibility_review"], "unknown")

    def test_camera_boundary_cases(self):
        cfg = config()
        validate_point([1, 1, 1], cfg, role="camera")
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_point([-0.01, 1, 1], cfg, role="camera")
        with self.assertRaisesRegex(ValueError, "expanded obstacle"):
            validate_point([3.75, 4, 1], cfg, role="camera")
        with self.assertRaisesRegex(ValueError, "expanded obstacle"):
            validate_point([4.5, 4.5, 1], cfg, role="camera")

    def test_old_captured_resume_cannot_bypass_new_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image=root/"rgb"; depth=root/"depth"
            image.write_bytes(b"i"); depth.write_bytes(b"d")
            row={"view_id":"v","status":"captured","split":"development","map_id":"complex",
                 "rgb_path":str(image),"depth_path":str(depth),"image_sha256":file_sha256(image),
                 "depth_sha256":file_sha256(depth)}
            receipt=root/"receipt.json"
            write_record(receipt,{"plan_identity":"p","mode":"calibration","map_id":"complex","views":[row]})
            with self.assertRaisesRegex(ValueError,"predates"):
                resumed_views(receipt,{"identity":"p","map_id":"complex"},"calibration",[view()],
                              config=config(),check_version=CHECK_VERSION,
                              instance_mapping={51:{"object_id":"transformer_a","category":"transformer"}})


if __name__ == "__main__":
    unittest.main()
