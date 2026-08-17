import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from src.inspection.config import InspectionConfig
from src.ml.artifacts import file_sha256, write_json
from src.sandbox.job_commands import build_command
from src.sandbox.workbench_models import WorkbenchDataset
from src.sandbox.workbench_recipe import materialize_workbench_recipe
from src.sandbox.workbench_lifecycle import inspect_workbench_run
from src.sandbox.workbench_runner import run_workbench_experiment
from src.sandbox.workbench_equivalence import calibration_images


class WorkbenchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.imported = self.root / "outputs/sandbox/workbench/datasets"
        self.runs = self.root / "outputs/sandbox/workbench/runs"
        self.dataset_root = self.imported / "dataset"
        for split in ("train", "validation"):
            (self.dataset_root / f"images/{split}").mkdir(parents=True)
            (self.dataset_root / f"labels/{split}").mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (16, 16), (20 + index, 30, 40)).save(
                    self.dataset_root / f"images/{split}/{split}-{index}.png"
                )
                (self.dataset_root / f"labels/{split}/{split}-{index}.txt").write_text(
                    "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
                )
        self.dataset = WorkbenchDataset(
            dataset_id="dataset", source_type="imported_yolo",
            dataset_root=str(self.dataset_root), dataset_identity_sha256="a" * 64,
            split_counts={"train": 2, "validation": 2},
        )
        write_json(self.dataset_root / "dataset.json", self.dataset.to_record())
        (self.root / "yolo11n.pt").write_bytes(b"weights")
        self.recipe, self.run_root = materialize_workbench_recipe(
            self.root, self.imported, self.runs, "experiment", "dataset", "smoke"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def completed_stages(self):
        best = self.run_root / "training/weights/best.pt"
        last = self.run_root / "training/weights/last.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        onnx = self.run_root / "model/model_320.onnx"
        onnx.parent.mkdir(parents=True)
        onnx.write_bytes(b"onnx")
        validation = {
            "confidence_evaluation": {"selected": {"threshold": 0.42}}
        }
        replay = {"replay_identity_sha256": "d" * 64}
        return best, last, onnx, validation, replay

    def test_runs_all_stages_and_writes_receipt(self):
        best, last, onnx, validation, replay = self.completed_stages()
        with patch("src.sandbox.workbench_runner._train", return_value=(best, last)), \
             patch("src.sandbox.workbench_runner.evaluate_workbench_model", return_value=(validation, [])), \
             patch("src.sandbox.workbench_runner._export_and_gate",
                   return_value=(onnx, [], {"passed": True})), \
             patch("src.sandbox.workbench_runner.replay_workbench_model", return_value=replay):
            write_json(self.run_root / "validation.json", validation)
            write_json(self.run_root / "onnx_equivalence.json", {"passed": True})
            receipt = run_workbench_experiment(
                self.root, self.imported, self.runs, self.run_root / "recipe.json"
            )
        self.assertTrue(receipt["passed"])
        inspected = inspect_workbench_run(self.run_root)
        self.assertEqual(inspected["status"]["state"], "complete")
        self.assertFalse(inspected["receipt"]["formal_evidence"])

    def test_epoch_progress_does_not_exceed_total(self):
        best, last, _, _, _ = self.completed_stages()

        class FakeModel:
            def add_callback(inner_self, _name, callback):
                inner_self.callback = callback

            def train(inner_self, **_kwargs):
                inner_self.callback(type("Trainer", (), {"epoch": 1})())

        module = types.ModuleType("ultralytics")
        module.YOLO = lambda _path: FakeModel()
        with patch.dict("sys.modules", {"ultralytics": module}):
            from src.sandbox.workbench_runner import _train
            _train(self.root / "yolo11n.pt", self.dataset_root,
                   self.run_root, self.recipe, False)
        status = json.loads((self.run_root / "status.json").read_text())
        self.assertEqual(status["epoch"], 1)
        self.assertEqual(status["total_epochs"], 1)

    def test_failure_is_persisted_with_stable_code(self):
        with patch("src.sandbox.workbench_runner._train",
                   side_effect=RuntimeError("workbench training requires requirements-ml.txt")):
            with self.assertRaises(RuntimeError):
                run_workbench_experiment(
                    self.root, self.imported, self.runs,
                    self.run_root / "recipe.json",
                )
        status = json.loads((self.run_root / "status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["failure_code"], "dependency_unavailable")

    def test_operator_builds_only_allowlisted_workbench_command(self):
        config = InspectionConfig(
            self.root, self.root / "plan.json", self.root / "collection",
            self.root / "px4", profile="development",
        )
        command = build_command(config, "workbench-run", parameters={
            "experiment_id": "operator-run", "dataset_id": "dataset",
            "preset": "smoke", "epochs": 1,
        })
        self.assertEqual(command.workflow, "visual_workbench")
        self.assertIn("workbench-run", command.argv)
        with self.assertRaisesRegex(ValueError, "unsupported workbench fields"):
            build_command(config, "workbench-run", parameters={
                "experiment_id": "bad", "dataset_id": "dataset",
                "preset": "smoke", "shell": "rm",
            })

        source = self.root / "external"
        source.mkdir()
        (source / "dataset.yaml").write_text("names: []\n", encoding="utf-8")
        imported = build_command(config, "workbench-dataset-import", parameters={
            "source": str(source), "dataset_id": "external-dataset",
            "class_map": {"3": "reactor", "0": "transformer",
                          "2": "capacitor_bank", "1": "switchgear"},
        })
        self.assertIn("0=transformer", imported.argv)
        with self.assertRaisesRegex(ValueError, "four distinct"):
            build_command(config, "workbench-dataset-import", parameters={
                "source": str(source), "dataset_id": "bad-map",
                "class_map": {"0": "transformer"},
            })

    def test_equivalence_calibration_is_bounded_and_class_stratified(self):
        validation = self.dataset_root / "images/validation"
        labels = self.dataset_root / "labels/validation"
        for index in range(250):
            image = validation / f"bulk-{index:03d}.png"
            Image.new("RGB", (16, 16), (index % 255, 10, 20)).save(image)
            class_id = index % 4
            (labels / f"bulk-{index:03d}.txt").write_text(
                f"{class_id} 0.5 0.5 0.4 0.4\n", encoding="utf-8"
            )
        selected = calibration_images(self.dataset_root, total=200)
        self.assertEqual(len(selected), 200)
        selected_classes = {
            int((labels / f"{path.stem}.txt").read_text().split()[0])
            for path in selected
        }
        self.assertEqual(selected_classes, {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
