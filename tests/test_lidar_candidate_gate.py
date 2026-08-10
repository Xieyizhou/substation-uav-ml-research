import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli.data import build_parser as build_data_parser
from src.cli.models import build_parser as build_model_parser
from src.ml.artifacts import write_json
from src.ml.candidate_gate import audit_candidate, run_validation_replay
from src.ml.model_package import create_model_package, validate_model_package
from src.ml.overlay_dataset import OverlaySource, materialize_overlay_dataset
from src.sensors.replay import append_scan_record
from src.sensors.types import LaserScanFrame


class LidarCandidateGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = []
        for name, map_id, seed in (
            ("train", "training", 2001),
            ("validation", "complex", 2041),
            ("test", "extreme", 2051),
        ):
            path = self.root / f"{name}.jsonl"
            for sequence in range(6):
                append_scan_record(path, self.frame(sequence))
            self.sources.append(OverlaySource(path, map_id, seed))

    def tearDown(self):
        self.temporary.cleanup()

    def frame(self, sequence):
        return LaserScanFrame(
            timestamp_s=float(sequence),
            received_monotonic_s=float(sequence),
            frame_id="lidar",
            angle_min_rad=-3.14,
            angle_max_rad=3.14,
            angle_step_rad=0.1,
            range_min_m=0.1,
            range_max_m=10.0,
            ranges_m=(5.0,) * 72,
            source="fixture",
            sequence=sequence,
        )

    def dataset(self, name="dataset"):
        output = self.root / name
        materialize_overlay_dataset(self.sources, output, max_frames_per_source=4)
        return output

    def package(self, dataset, name="package"):
        source = self.root / f"{name}.onnx"
        source.write_bytes(b"model")
        history = self.root / f"{name}-history.json"
        metrics = self.root / f"{name}-metrics.json"
        history.write_text("[]", encoding="utf-8")
        per_class = {
            label: {"support": 4, "recall": 0.9}
            for label in ("clear", "warning", "danger")
        }
        write_json(
            metrics,
            {
                "validation": {"macro_f1": 0.9, "per_class": per_class},
                "test": {"macro_f1": 0.9, "per_class": per_class},
                "onnx": {
                    "max_absolute_error": 1e-6,
                    "cpu_latency": {"p95_ms": 1.0},
                },
            },
        )
        package = self.root / name
        create_model_package(
            package,
            onnx_path=source,
            dataset_manifest_path=dataset / "dataset_manifest.json",
            training_history_path=history,
            offline_metrics_path=metrics,
            model_id=name,
            training_parameters={"allow_incomplete_labels": False},
        )
        manifest_path = package / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["training_commit"] = "a" * 40
        write_json(manifest_path, manifest)
        return package

    def test_overlay_is_balanced_split_safe_and_deterministic(self):
        first = self.dataset("first")
        second = self.dataset("second")
        left = json.loads((first / "dataset_manifest.json").read_text())
        right = json.loads((second / "dataset_manifest.json").read_text())
        self.assertEqual(left["dataset_id"], right["dataset_id"])
        self.assertEqual(left["split_counts"], {"test": 12, "train": 12, "validation": 12})
        self.assertEqual(
            left["risk_label_distribution"],
            {"clear": 12, "danger": 12, "warning": 12},
        )
        self.assertEqual(left["overlay_version"], "deterministic-risk-overlay-v1")
        with self.assertRaisesRegex(ValueError, "already exists"):
            materialize_overlay_dataset(self.sources, first)

    def test_cli_registers_overlay_readiness_and_replay_gate(self):
        overlay = build_data_parser().parse_args(
            ["overlay", "--source", "scan.jsonl:simple:2001", "--output", "out"]
        )
        self.assertEqual(overlay.command, "overlay")
        readiness = build_model_parser().parse_args(
            ["readiness", "--package", "model", "--dataset", "dataset"]
        )
        self.assertEqual(readiness.command, "readiness")
        replay = build_model_parser().parse_args(
            [
                "replay-gate", "--package", "model", "--dataset", "dataset",
                "--output", "gate",
            ]
        )
        self.assertEqual(replay.command, "replay-gate")

    def test_readiness_rejects_dirty_or_incomplete_candidate(self):
        dataset = self.dataset()
        package = self.package(dataset)
        manifest_path = package / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["training_commit"] = "abc-dirty"
        manifest["training_parameters"]["allow_incomplete_labels"] = True
        write_json(manifest_path, manifest)
        result = audit_candidate(package, dataset)
        failed = {row["name"] for row in result["checks"] if not row["passed"]}
        self.assertEqual(
            failed, {"clean_training_commit", "complete_training_labels"}
        )

    def test_external_onnx_data_is_copied_hashed_and_validated(self):
        dataset = self.dataset()
        source = self.root / "external.onnx"
        source.write_bytes(b"graph")
        source.with_suffix(".onnx.data").write_bytes(b"weights")
        history, metrics = self.root / "h.json", self.root / "m.json"
        history.write_text("[]")
        metrics.write_text("{}")
        package = self.root / "external-package"
        manifest = create_model_package(
            package,
            onnx_path=source,
            dataset_manifest_path=dataset / "dataset_manifest.json",
            training_history_path=history,
            offline_metrics_path=metrics,
        )
        self.assertTrue(manifest["onnx_external_data_sha256"])
        self.assertTrue((package / "model.onnx.data").is_file())
        (package / "model.onnx.data").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "external data hash mismatch"):
            validate_model_package(package)

    @patch("src.ml.candidate_gate.evaluate_predictions")
    @patch("src.ml.candidate_gate.predict_dataset")
    def test_validation_replay_writes_identity_bound_gate(self, predict, evaluate):
        dataset = self.dataset()
        package = self.package(dataset)

        def write_predictions(model, samples, output, split):
            self.assertEqual(split, "validation")
            Path(output).write_text('{"prediction":1}\n', encoding="utf-8")
            return Path(output)

        predict.side_effect = write_predictions
        evaluate.return_value = {
            "macro_f1": 0.8,
            "per_class": {
                "clear": {"recall": 0.9},
                "warning": {"recall": 0.8},
                "danger": {"recall": 0.7},
            },
            "traversability_iou": 0.75,
            "ece": 0.05,
            "latency": {"p95_ms": 2.0},
        }
        result = run_validation_replay(package, dataset, self.root / "gate")
        self.assertTrue(result["passed"])
        self.assertEqual(result["partition"], "validation")
        self.assertTrue(result["replay_gate_identity_sha256"])
        self.assertTrue((self.root / "gate/replay_gate.json").is_file())


if __name__ == "__main__":
    unittest.main()
