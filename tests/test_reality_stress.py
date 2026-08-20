import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from src.vision.evaluation.reality_stress import materialize_reality_stress


class RealityStressTests(unittest.TestCase):
    def test_rejects_too_small_view(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "images"
            source.mkdir()
            Image.new("RGB", (20, 10)).save(source / "one.jpg")
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps({
                "one": {"filename": "one.jpg", "regions": []}
            }))
            with self.assertRaisesRegex(ValueError, "300 to 1000"):
                materialize_reality_stress(source, annotation, root / "output")

    def test_materializes_mapped_and_empty_labels_deterministically(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "images"
            source.mkdir()
            entries = {}
            for index in range(300):
                filename = f"{index:03}.jpg"
                Image.new("RGB", (20, 10)).save(source / filename)
                source_type = "switch" if index == 0 else "insulator"
                entries[str(index)] = {
                    "filename": filename,
                    "regions": [{
                        "shape_attributes": {
                            "all_points_x": [2, 10, 10, 2],
                            "all_points_y": [1, 1, 8, 8],
                        },
                        "region_attributes": {"type": source_type},
                    }],
                }
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps(entries))
            result = materialize_reality_stress(
                source, annotation, root / "output"
            )
            self.assertEqual(result["frame_count"], 300)
            self.assertEqual(result["mapped_object_counts"], {"switchgear": 1})
            self.assertEqual(
                (root / "output/labels/stress/000.txt").read_text().split()[0],
                "1",
            )
            self.assertEqual(
                (root / "output/labels/stress/001.txt").read_text(), ""
            )
            self.assertEqual(
                result["unsupported_target_classes"],
                ["transformer", "capacitor_bank", "reactor"],
            )


if __name__ == "__main__":
    unittest.main()
