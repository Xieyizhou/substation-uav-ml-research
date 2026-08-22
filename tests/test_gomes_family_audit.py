import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from src.vision.training.gomes_family_audit import audit_gomes_source_family


class GomesFamilyAuditTests(unittest.TestCase):
    def test_reports_cross_release_overlap_and_pending_unique_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yolo = root / "yolo"
            yolo.mkdir()
            classes = yolo / "classes.txt"
            classes.write_text("Other\nPower transformer\n", encoding="utf-8")
            for index, name in enumerate((
                "misc.zip", "agv_day.zip", "agv_night_light.zip", "agv_night_dark.zip"
            )):
                with zipfile.ZipFile(yolo / name, "w") as bundle:
                    image = _jpeg_bytes(10 if index < 2 else 40 + index)
                    bundle.writestr(f"images/item-{index}.jpg", image)
                    bundle.writestr(
                        f"labels/item-{index}.txt",
                        "1 0.5 0.5 0.2 0.2\n" if index != 3 else "0 0.5 0.5 0.2 0.2\n",
                    )
            semantic = root / "images.zip"
            with zipfile.ZipFile(semantic, "w") as bundle:
                bundle.writestr("images/a.jpg", _jpeg_bytes(10))
                bundle.writestr("images/b.jpg", _jpeg_bytes(90))

            result = audit_gomes_source_family(semantic, yolo, classes, root / "out")

            self.assertEqual(result["cross_release_exact_cluster_count"], 1)
            self.assertEqual(result["semantic_images_present_in_yolo_count"], 1)
            self.assertIn("cross_release_near_overlap_candidate_count", result)
            self.assertEqual(result["transformer_candidate_independent_image_count"], 2)
            self.assertEqual(result["accepted_transformer_image_count"], 0)
            self.assertEqual(result["semantic_release_unreadable_member_count"], 0)
            rows = [json.loads(line) for line in
                    (root / "out" / "transformer-review.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["review_status"] == "pending" for row in rows))
            self.assertFalse(result["training_eligible"])


if __name__ == "__main__":
    unittest.main()


def _jpeg_bytes(value):
    output = BytesIO()
    Image.new("L", (12, 12), color=value).save(output, format="JPEG")
    return output.getvalue()
