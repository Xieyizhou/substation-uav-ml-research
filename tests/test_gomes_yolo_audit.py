from hashlib import md5
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.vision.training.gomes_yolo_audit import audit_gomes_yolo_component


class GomesYoloAuditTests(unittest.TestCase):
    def test_audits_atomic_capture_group_and_transformer_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classes = root / "classes.txt"
            classes.write_text("Other\nPower transformer\n", encoding="utf-8")
            archive = root / "capture.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("capture/a.jpg", b"image")
                bundle.writestr("capture/labels/a.txt", "1 0.5 0.5 0.2 0.2\n")
            record = {
                "real_domain_source_registry_schema_version": 2,
                "sources": [{
                    "source_id": "gomes-test", "name": "Gomes", "version": "v1",
                    "landing_url": "https://example.test", "download_url": "https://example.test",
                    "declared_license_id": "CC-BY-4.0",
                    "declared_license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "upstream_url": "https://example.test", "upstream_license_id": "CC-BY-4.0",
                    "citation": "Gomes", "source_family": "gomes-site", "review_priority": 1,
                    "provenance_status": "verified", "ingestion_status": "quarantine",
                    "usage_role": "development_candidate", "eligible_partitions": [],
                    "annotation_format": "yolo-detection-capture-groups", "site_group_count": 1,
                    "reported_image_count": 1,
                    "class_candidates": {"transformer": {"source_labels": ["Power transformer"], "semantic_status": "verified"}},
                    "artifact_components": [
                        {"filename": "capture.zip", "bytes": archive.stat().st_size,
                         "md5": md5(archive.read_bytes(), usedforsecurity=False).hexdigest(), "capture_group": "capture"},
                        {"filename": "classes.txt", "bytes": classes.stat().st_size,
                         "md5": md5(classes.read_bytes(), usedforsecurity=False).hexdigest(), "capture_group": "metadata"}
                    ], "blocking_issues": ["atomic site"]
                }]
            }
            registry = root / "registry.json"
            registry.write_text(json.dumps(record), encoding="utf-8")
            receipt = audit_gomes_yolo_component(registry, "gomes-test", archive, classes, root / "out")
            self.assertEqual(receipt["transformer_image_count"], 1)
            self.assertEqual(receipt["transformer_annotation_count"], 1)
            self.assertEqual(receipt["eligible_partition_after_family_review"], "development")
            self.assertFalse(receipt["training_eligible"])


if __name__ == "__main__":
    unittest.main()
