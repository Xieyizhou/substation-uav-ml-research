import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from src.vision.training.source_content_audit import audit_source_content


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config/perception/real_domain_sources.json"


class SourceContentAuditTests(unittest.TestCase):
    def _png(self, color):
        with tempfile.NamedTemporaryFile(suffix=".png") as target:
            Image.new("RGB", (16, 16), color).save(target.name)
            return Path(target.name).read_bytes()

    def _archive(self, root, duplicate=False):
        archive = root / "source.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("data.yaml", "names: [Reactor]\n")
            bundle.writestr("README.roboflow.txt", """dataset - v1\nThe dataset includes 2 images.\nThe following pre-processing was applied to each image:\n* Resize to 640x640\nNo image augmentation techniques were applied.\n""")
            image = self._png((10, 20, 30))
            bundle.writestr("train/images/a.rf.aaa.png", image)
            bundle.writestr("train/labels/a.rf.aaa.txt", "0 0.5 0.5 0.2 0.2\n")
            bundle.writestr("valid/images/a.rf.bbb.png", image if duplicate else self._png((40, 50, 60)))
            bundle.writestr("valid/labels/a.rf.bbb.txt", "0 0.5 0.5 0.2 0.2\n")
        return archive

    def test_streams_images_and_reports_lineage_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = audit_source_content(
                REGISTRY, "space-weather-reactors", self._archive(root, True), root / "out"
            )
            self.assertEqual(receipt["independent_image_count"], 1)
            self.assertEqual(receipt["cross_split_leakage_count"], 1)
            self.assertEqual(receipt["lineage_cluster_count"], 1)
            self.assertEqual(receipt["candidate_unique_image_count"]["reactor"], 1)
            self.assertTrue(receipt["augmentation_evidence_detected"])

    def test_receipt_is_deterministic_and_keeps_review_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            first = audit_source_content(REGISTRY, "space-weather-reactors", archive, root / "one")
            second = audit_source_content(REGISTRY, "space-weather-reactors", archive, root / "two")
            self.assertEqual(first["content_audit_identity_sha256"], second["content_audit_identity_sha256"])
            self.assertTrue(first["semantic_review_manifest"]["ambiguous"])
            self.assertFalse(first["training_eligible"])


if __name__ == "__main__":
    unittest.main()
