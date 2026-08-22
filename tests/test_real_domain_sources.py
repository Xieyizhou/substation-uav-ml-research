import json
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

from PIL import Image

from src.cli.visual import build_parser
from src.vision.training.real_domain_dataset import audit_real_dataset
from src.ml.artifacts import file_sha256
from src.vision.evaluation.yolo_package_v3 import bind_v3_gate, validate_v3_gate
from src.vision.training.source_feasibility import (
    audit_real_sources,
    load_source_registry,
)
from src.vision.training.source_intake import intake_real_source
from src.vision.training.source_export_audit import audit_source_export
from src.vision.training.source_download import download_approved_source


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config/perception/real_domain_sources.json"


class RealDomainSourceTests(unittest.TestCase):
    def test_repository_registry_never_approves_unverified_sources(self):
        _, sources = load_source_registry(REGISTRY)
        self.assertEqual(
            sources["gomes-substation-equipment-zenodo-7884270"][
                "ingestion_status"
            ],
            "approved",
        )
        for source_id in (
            "space-weather-reactors",
            "roboflow-substation-equipment-det-2-v2",
            "fin-switch-gear",
            "electrical-devices-vit-atwcp",
        ):
            self.assertEqual(sources[source_id]["ingestion_status"], "quarantine")
        rejected_space = sources["space-weather-merged-power"]
        self.assertEqual(rejected_space["ingestion_status"], "rejected")
        self.assertEqual(rejected_space["usage_role"], "evaluation_only")
        self.assertEqual(rejected_space["eligible_partitions"], ["stress"])
        self.assertEqual(sources["fin-switch-gear"]["access"]["dataset_locator"], "sankets-workspace-qdfcy/fin-switch-gear/2")
        self.assertEqual(sources["roboflow-substation-equipment-34"]["ingestion_status"], "rejected")
        self.assertEqual(sources["roboflow-substation-equipment-34"]["usage_role"], "evaluation_only")
        stress = sources["power-equipment-image-dataset-mit"]
        self.assertEqual(stress["usage_role"], "evaluation_only")
        self.assertEqual(stress["eligible_partitions"], ["stress"])

    def test_feasibility_report_recommends_audit_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            report = audit_real_sources(REGISTRY, directory)
            self.assertFalse(report["ready_to_materialize_training_dataset"])
            self.assertEqual(
                report["verified_training_sources_by_class"]["transformer"],
                ["gomes-substation-equipment-zenodo-7884270"],
            )
            self.assertEqual(
                set(report["missing_verified_classes"]),
                {"switchgear", "capacitor_bank", "reactor"},
            )
            self.assertEqual(
                report["recommended_strategy"]["decision"],
                "hold_training_and_audit_quarantine_sources",
            )
            self.assertEqual(
                report["recommended_strategy"]["audit_next"][:3],
                [
                    "space-weather-reactors",
                    "fin-switch-gear",
                    "roboflow-substation-equipment-det-2-v2",
                ],
            )
            self.assertEqual(len(report["source_partition_gaps"]), 11)
            self.assertTrue(
                (Path(directory) / "real_source_feasibility.json").is_file()
            )
            self.assertTrue(
                (Path(directory) / "real_source_feasibility.csv").is_file()
            )

    def test_approved_source_requires_verified_upstream_rights(self):
        record = json.loads(REGISTRY.read_text())
        candidate = record["sources"][1]
        candidate["ingestion_status"] = "approved"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified provenance"):
                load_source_registry(path)

    def test_dataset_manifest_rejects_quarantine_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (16, 16), (0, 0, 0)).save(image_path)
            manifest = root / "samples.jsonl"
            manifest.write_text(json.dumps({
                "sample_id": "sample-1",
                "source_id": "space-weather-reactors",
                "partition": "development",
                "group_id": "site-1",
                "image_relative_path": "sample.png",
                "annotations": [{
                    "class_name": "reactor",
                    "bbox_xyxy": [1, 1, 12, 12],
                }],
            }) + "\n", encoding="utf-8")
            result = audit_real_dataset(REGISTRY, manifest, root)
            self.assertFalse(result["valid"])
            self.assertIn("not approved", result["errors"][0]["error"])

    def test_dataset_manifest_rejects_unverified_source_class(self):
        result = self._audit_gomes_sample("development", "switchgear")
        self.assertFalse(result["valid"])
        self.assertIn("semantics are not verified", result["errors"][0]["error"])

    def test_dataset_manifest_rejects_disallowed_source_partition(self):
        result = self._audit_gomes_sample("validation", "transformer")
        self.assertFalse(result["valid"])
        self.assertIn("requested partition", result["errors"][0]["error"])

    def test_dataset_audit_rejects_cross_source_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = json.loads(REGISTRY.read_text())
            dup = dict(next(
                source for source in record["sources"]
                if source["source_id"] == "gomes-substation-equipment-zenodo-7884270"
            ))
            dup["source_id"] = "gomes-substation-equipment-zenodo-7884270-dup"
            record["sources"].append(dup)
            registry = root / "registry.json"
            registry.write_text(json.dumps(record), encoding="utf-8")
            Image.new("RGB", (16, 16), (0, 0, 0)).save(root / "sample.png")
            Image.new("RGB", (16, 16), (0, 0, 0)).save(root / "sample-copy.png")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps({
                    "sample_id": "sample-1",
                    "source_id": "gomes-substation-equipment-zenodo-7884270",
                    "partition": "development",
                    "group_id": "site-a",
                    "image_relative_path": "sample.png",
                    "annotations": [{
                        "class_name": "transformer",
                        "bbox_xyxy": [1, 1, 12, 12],
                    }],
                }) + "\n" + json.dumps({
                    "sample_id": "sample-2",
                    "source_id": "gomes-substation-equipment-zenodo-7884270-dup",
                    "partition": "development",
                    "group_id": "site-b",
                    "image_relative_path": "sample-copy.png",
                    "annotations": [{
                        "class_name": "transformer",
                        "bbox_xyxy": [1, 1, 12, 12],
                    }],
                }) + "\n", encoding="utf-8"
            )
            result = audit_real_dataset(registry, manifest, root)
            self.assertFalse(result["valid"])
            self.assertEqual(result["cross_source_duplicate_count"], 1)
            self.assertEqual(result["exact_duplicate_count"], 1)

    def test_dataset_audit_rejects_site_split_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "samples.jsonl"
            manifest.write_text(json.dumps({
                "sample_id": "sample-1",
                "source_id": "gomes-substation-equipment-zenodo-7884270",
                "partition": "development",
                "group_id": "same-site",
                "image_relative_path": "sample.png",
                "annotations": [{
                    "class_name": "transformer",
                    "bbox_xyxy": [1, 1, 12, 12],
                }],
            }) + "\n" + json.dumps({
                "sample_id": "sample-2",
                "source_id": "gomes-substation-equipment-zenodo-7884270",
                "partition": "validation",
                "group_id": "same-site",
                "image_relative_path": "sample2.png",
                "annotations": [{
                    "class_name": "transformer",
                    "bbox_xyxy": [1, 1, 12, 12],
                }],
            }) + "\n", encoding="utf-8")
            Image.new("RGB", (16, 16), (0, 0, 0)).save(root / "sample.png")
            Image.new("RGB", (16, 16), (10, 10, 10)).save(root / "sample2.png")
            result = audit_real_dataset(REGISTRY, manifest, root)
            self.assertFalse(result["valid"])
            self.assertIn("site or sequence group crosses dataset splits", result["errors"][0]["error"])

    def _audit_gomes_sample(self, partition, class_name):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        Image.new("RGB", (16, 16), (0, 0, 0)).save(root / "sample.png")
        manifest = root / "samples.jsonl"
        manifest.write_text(json.dumps({
            "sample_id": "sample-1",
            "source_id": "gomes-substation-equipment-zenodo-7884270",
            "partition": partition,
            "group_id": "gomes-site-1",
            "image_relative_path": "sample.png",
            "annotations": [{
                "class_name": class_name,
                "bbox_xyxy": [1, 1, 12, 12],
            }],
        }) + "\n", encoding="utf-8")
        return audit_real_dataset(REGISTRY, manifest, root)

    def test_cli_exposes_source_feasibility_audit(self):
        args = build_parser().parse_args(["real-source-audit"])
        self.assertEqual(args.command, "real-source-audit")
        self.assertEqual(args.registry, Path("config/perception/real_domain_sources.json"))

    def test_cli_exposes_bounded_source_intake(self):
        args = build_parser().parse_args([
            "real-source-intake", "--source-id", "space-weather-reactors",
        ])
        self.assertEqual(args.command, "real-source-intake")
        self.assertEqual(args.sample_limit, 8)

    def test_cli_exposes_manual_export_audit(self):
        args = build_parser().parse_args([
            "real-source-export-audit",
            "--source-id", "space-weather-reactors",
            "--archive", "reactors.zip",
        ])
        self.assertEqual(args.command, "real-source-export-audit")
        self.assertEqual(args.archive, Path("reactors.zip"))

    def test_cli_exposes_approved_source_download(self):
        args = build_parser().parse_args([
            "real-source-download",
            "--source-id", "gomes-substation-equipment-zenodo-7884270",
        ])
        self.assertEqual(args.command, "real-source-download")

    def test_source_intake_without_previews_requires_manual_export(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = intake_real_source(
                REGISTRY, "space-weather-reactors", directory,
            )
            self.assertEqual(receipt["sample_count"], 0)
            self.assertTrue(receipt["manual_export_required"])
            self.assertFalse(receipt["training_eligible"])
            self.assertTrue((
                Path(directory) / "space-weather-reactors" /
                "intake_receipt.json"
            ).is_file())

    def test_source_intake_rejects_approved_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "quarantine"):
                intake_real_source(
                    REGISTRY,
                    "gomes-substation-equipment-zenodo-7884270",
                    directory,
                )

    def test_source_export_audit_accepts_structural_yolo_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "reactors.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data.yaml", "names: [Reactor]\n")
                bundle.writestr("train/images/a.jpg", b"image")
                bundle.writestr("train/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
                bundle.writestr("valid/images/b.jpg", b"image")
                bundle.writestr("valid/labels/b.txt", "")
            receipt = audit_source_export(
                REGISTRY, "space-weather-reactors", archive, root / "audit",
            )
            self.assertTrue(receipt["structurally_valid"])
            self.assertFalse(receipt["training_eligible"])
            self.assertEqual(receipt["split_image_counts"]["train"], 1)
            self.assertEqual(receipt["split_image_counts"]["validation"], 1)

    def test_source_export_audit_accepts_duplicate_yaml_when_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "duplicate.yaml.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("train/data.yaml", "names: [Reactor]\n")
                bundle.writestr("extra/data.yaml", "names: [Reactor]\n")
                bundle.writestr("train/images/a.jpg", b"image")
                bundle.writestr("train/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
                bundle.writestr("valid/images/b.jpg", b"image")
                bundle.writestr("valid/labels/b.txt", "")
            receipt = audit_source_export(
                REGISTRY, "space-weather-reactors", archive, root / "audit",
            )
            self.assertTrue(receipt["structurally_valid"])
            self.assertEqual(receipt["duplicate_entry_count"], 1)
            self.assertIn("byte-identical", receipt["data_yaml_warnings"][0])

    def test_source_export_audit_rejects_conflicting_duplicate_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "conflict.yaml.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data.yaml", "names: [Reactor]\n")
                bundle.writestr("train/data.yaml", "names: [NotReactor]\n")
                bundle.writestr("train/images/a.jpg", b"image")
                bundle.writestr("train/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
                bundle.writestr("valid/images/b.jpg", b"image")
                bundle.writestr("valid/labels/b.txt", "")
            with self.assertRaisesRegex(ValueError, "conflicting content"):
                audit_source_export(
                    REGISTRY, "space-weather-reactors", archive, root / "audit",
                )

    def test_source_export_audit_accepts_polygon_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "det2.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data.yaml", "names: [main transformer]\n")
                bundle.writestr("train/images/a.jpg", b"image")
                bundle.writestr(
                    "train/labels/a.txt",
                    "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
                )
                bundle.writestr("valid/images/b.jpg", b"image")
                bundle.writestr("valid/labels/b.txt", "")
            receipt = audit_source_export(
                REGISTRY, "roboflow-substation-equipment-det-2-v2", archive, root / "audit",
            )
            self.assertTrue(receipt["structurally_valid"])
            self.assertEqual(receipt["segmentation_annotation_count"], 1)
            self.assertEqual(receipt["detection_annotation_count"], 0)
            self.assertEqual(receipt["annotation_conversion"], ["polygon_to_xyxy_bbox_v1"])
            self.assertEqual(receipt["class_annotation_count"]["main transformer"], 1)
            self.assertEqual(receipt["class_image_count"]["main transformer"], 1)

    def test_source_export_audit_accepts_same_input_with_stable_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "stable.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data.yaml", "names: [Reactor]\n")
                bundle.writestr("train/images/a.jpg", b"image")
                bundle.writestr("train/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
                bundle.writestr("valid/images/b.jpg", b"image")
                bundle.writestr("valid/labels/b.txt", "")
            first = audit_source_export(
                REGISTRY, "space-weather-reactors", archive, root / "run1",
            )
            second = audit_source_export(
                REGISTRY, "space-weather-reactors", archive, root / "run2",
            )
            self.assertEqual(
                first["export_audit_identity_sha256"],
                second["export_audit_identity_sha256"],
            )

    def test_source_export_audit_rejects_invalid_polygon_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "invalid-seg.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data.yaml", "names: [Reactor]\n")
                bundle.writestr("train/images/a.jpg", b"image")
                bundle.writestr("train/labels/a.txt", "0 0.1 0.1 0.9 0.1 0.5\n")
                bundle.writestr("valid/images/b.jpg", b"image")
                bundle.writestr("valid/labels/b.txt", "")
            receipt = audit_source_export(
                REGISTRY, "space-weather-reactors", archive, root / "audit",
            )
            self.assertFalse(receipt["structurally_valid"])
            self.assertEqual(receipt["invalid_label_file_count"], 1)

    def test_source_export_audit_rejects_unsafe_zip_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data.yaml", "names: [Reactor]\n")
                bundle.writestr("../outside.jpg", b"image")
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                audit_source_export(
                    REGISTRY, "space-weather-reactors", archive, root / "audit",
                )

    def test_source_export_audit_reports_invalid_yolo_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "invalid-label.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("data.yaml", "names: [Reactor]\n")
                bundle.writestr("train/images/a.jpg", b"image")
                bundle.writestr("train/labels/a.txt", "1 0.5 0.5 0.2 0.2\n")
                bundle.writestr("valid/images/b.jpg", b"image")
                bundle.writestr("valid/labels/b.txt", "")
            receipt = audit_source_export(
                REGISTRY, "space-weather-reactors", archive, root / "audit",
            )
            self.assertFalse(receipt["structurally_valid"])
            self.assertEqual(receipt["invalid_label_file_count"], 1)

    def test_approved_source_download_verifies_pinned_artifact(self):
        payload = b"licensed-source"
        record = json.loads(REGISTRY.read_text())
        source = record["sources"][0]
        source["public_artifact"] = {
            "url": "https://zenodo.org/test.zip",
            "filename": "test.zip",
            "bytes": len(payload),
            "upstream_checksum": "md5:" + hashlib.md5(
                payload, usedforsecurity=False
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            registry.write_text(json.dumps(record), encoding="utf-8")
            response = _FakeResponse(payload)
            from unittest.mock import patch
            with patch("urllib.request.urlopen", return_value=response):
                receipt = download_approved_source(
                    registry, source["source_id"], root / "downloads",
                )
            self.assertEqual(receipt["bytes"], len(payload))
            self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())

    def test_v3_package_gate_is_copied_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            weights.write_bytes(b"weights")
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps({
                "passed": True,
                "model_sha256": file_sha256(weights),
                "training_view_identity_sha256": "view-1",
            }), encoding="utf-8")
            package = root / "package"
            package.mkdir()
            reference = bind_v3_gate(
                gate_path,
                weights,
                SimpleNamespace(training_view_identity_sha256="view-1"),
                package,
            )
            validate_v3_gate(package, reference)

    def test_v3_package_gate_rejects_other_training_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            weights.write_bytes(b"weights")
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps({
                "passed": True,
                "model_sha256": file_sha256(weights),
                "training_view_identity_sha256": "other-view",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another training view"):
                bind_v3_gate(
                    gate_path,
                    weights,
                    SimpleNamespace(training_view_identity_sha256="view-1"),
                    root / "package",
                )


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    unittest.main()
