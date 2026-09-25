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
from src.sandbox.workbench_lifecycle import (
    inspect_workbench_run, replay_workbench_run,
)
from src.sandbox.workbench_run_guard import exclusive_workbench_run
from src.sandbox.workbench_inference import run_image_inference
from src.sandbox.workbench_receipt import materialize_workbench_receipt
from src.sandbox.workbench_runner import _prepare_view, run_workbench_experiment
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
                    "".join(f"{class_id} 0.5 0.5 0.4 0.4\n" for class_id in range(4)), encoding="utf-8"
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
        replay = {
            "model_sha256": file_sha256(onnx),
            "replay_identity_sha256": "d" * 64,
        }
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
        # Keep native torch out of the temporary sys.modules snapshot. Removing
        # and re-importing its C extension can crash subsequent tests on macOS.
        with patch.dict("sys.modules", {"ultralytics": module}), patch(
            "src.sandbox.workbench_runner._device", return_value="cpu"
        ):
            from src.sandbox.workbench_runner import _train
            _train(self.root / "yolo11n.pt", self.dataset_root,
                   self.run_root, self.recipe, False)
        status = json.loads((self.run_root / "status.json").read_text())
        self.assertEqual(status["epoch"], 1)
        self.assertEqual(status["total_epochs"], 1)

    def test_view_preparation_is_idempotent_for_existing_hardlinks(self):
        parameters = self.recipe.parameters
        view, first = _prepare_view(self.dataset, self.run_root, parameters)
        second_view, second = _prepare_view(self.dataset, self.run_root, parameters)
        self.assertEqual(second_view, view)
        self.assertEqual(first["membership_sha256"], second["membership_sha256"])
        self.assertEqual(second["link_modes"], {"existing": 4})

    def test_missing_validation_class_is_rejected_before_training(self):
        for path in (self.dataset_root / "labels/validation").glob("*.txt"):
            path.write_text("0 0.5 0.5 0.4 0.4\n")
        with patch("src.sandbox.workbench_runner._train") as train:
            with self.assertRaisesRegex(ValueError, "validation selection lacks required classes"):
                run_workbench_experiment(self.root, self.imported, self.runs,
                                         self.run_root / "recipe.json")
        train.assert_not_called()

    def test_completed_run_is_not_restarted_and_stale_status_is_repaired(self):
        best, last, onnx, validation, replay = self.completed_stages()
        with patch("src.sandbox.workbench_runner._train", return_value=(best, last)), \
             patch("src.sandbox.workbench_runner.evaluate_workbench_model", return_value=(validation, [])), \
             patch("src.sandbox.workbench_runner._export_and_gate",
                   return_value=(onnx, [], {"passed": True})), \
             patch("src.sandbox.workbench_runner.replay_workbench_model", return_value=replay):
            write_json(self.run_root / "validation.json", validation)
            write_json(self.run_root / "onnx_equivalence.json", {"passed": True})
            write_json(self.run_root / "replay.json", replay)
            expected = run_workbench_experiment(
                self.root, self.imported, self.runs, self.run_root / "recipe.json"
            )
        write_json(self.run_root / "status.json", {"state": "running", "stage": "replay"})
        with patch("src.sandbox.workbench_runner._prepare_view") as prepare:
            actual = run_workbench_experiment(
                self.root, self.imported, self.runs,
                self.run_root / "recipe.json", resume=True,
            )
        prepare.assert_not_called()
        self.assertEqual(actual, expected)
        status = json.loads((self.run_root / "status.json").read_text())
        self.assertEqual((status["state"], status["stage"]), ("complete", "complete"))

    def test_run_directory_rejects_a_second_process_owner(self):
        with exclusive_workbench_run(self.run_root):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with exclusive_workbench_run(self.run_root):
                    self.fail("duplicate workbench owner was accepted")

    def test_replay_refreshes_the_final_receipt(self):
        best, _, onnx, validation, replay = self.completed_stages()
        write_json(self.run_root / "validation.json", validation)
        write_json(self.run_root / "onnx_equivalence.json", {"passed": True})
        comparison = {"status": "complete"}
        with patch(
            "src.sandbox.workbench_lifecycle.replay_workbench_model",
            return_value=replay,
        ), patch(
            "src.sandbox.workbench_lifecycle.compare_with_baseline",
            return_value=comparison,
        ):
            result = replay_workbench_run(self.root, self.run_root)
        self.assertEqual(result["receipt"]["onnx_model_sha256"], file_sha256(onnx))
        status = json.loads((self.run_root / "status.json").read_text())
        self.assertEqual((status["state"], status["stage"]), ("complete", "complete"))

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

    def _verified_run(self):
        best, _, onnx, validation, replay = self.completed_stages()
        write_json(self.run_root / "validation.json", validation)
        gate = {"passed": True}
        write_json(self.run_root / "onnx_equivalence.json", gate)
        write_json(self.run_root / "replay.json", replay)
        materialize_workbench_receipt(
            self.root, self.run_root, self.recipe, best, onnx, gate, replay,
            {"status": "complete"},
        )

    def test_verified_local_image_inference_writes_identity_bound_result(self):
        self._verified_run()
        source = self.root / "source.png"
        Image.new("RGB", (80, 60), "white").save(source)

        def predict(_model, _image):
            return ([{
                "class_id": 0, "class_name": "transformer", "confidence": 0.9,
                "xyxy_pixels": [5.0, 5.0, 40.0, 35.0],
            }], 12.5)

        output = self.root / "inference/inference-1"
        result = run_image_inference(
            self.runs, self.recipe.experiment_id, source, output, predictor=predict
        )
        self.assertFalse(result["formal_evidence"])
        self.assertEqual(result["results"]["primary"]["detection_count"], 1)
        self.assertEqual(result["results"]["primary"]["threshold"], 0.42)
        self.assertTrue((output / "primary.png").is_file())
        self.assertTrue((output / "result.json").is_file())

    def test_operator_finetunes_verified_parent_and_pins_its_identity(self):
        self._verified_run()
        config = InspectionConfig(
            self.root, self.root / "plan.json", self.root / "collection",
            self.root / "px4", profile="development",
        )
        build_command(config, "workbench-run", parameters={
            "experiment_id": "child", "dataset_id": "dataset", "preset": "smoke",
            "parent_experiment_id": self.recipe.experiment_id,
        })
        from src.sandbox.workbench_models import WorkbenchExperimentRecipe
        from src.sandbox.workbench_weights import resolve_initial_weights
        child = WorkbenchExperimentRecipe.from_record(json.loads(
            (self.runs / "child/recipe.json").read_text()
        ))
        self.assertEqual(child.initialization["parent_experiment_id"], "experiment")
        self.assertEqual(child.initialization["parent_threshold"], 0.42)
        self.assertEqual(resolve_initial_weights(self.root, child).read_bytes(), b"best")
        # The new experiment remains reproducible after the original is edited.
        (self.run_root / "training/weights/best.pt").write_bytes(b"changed")
        self.assertEqual(resolve_initial_weights(self.root, child).read_bytes(), b"best")
        with self.assertRaisesRegex(ValueError, "completed artifact changed"):
            materialize_workbench_recipe(
                self.root, self.imported, self.runs, "rejected-child", "dataset", "smoke",
                parent_experiment_id="experiment",
            )

    def test_new_receipt_binds_comparison_and_replay_bytes(self):
        from src.sandbox.workbench_run_guard import verified_completed_receipt
        comparison = {"status": "complete", "deltas": {"macro_f1": -0.1}}
        write_json(self.run_root / "comparison.json", comparison)
        self._verified_run()
        receipt = verified_completed_receipt(self.run_root, self.recipe)
        self.assertEqual(receipt["comparison_sha256"], file_sha256(self.run_root / "comparison.json"))
        for filename in ("comparison.json", "replay.json"):
            path = self.run_root / filename
            original = path.read_bytes()
            value = json.loads(original)
            value["unbound_mutation"] = True
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "completed artifact changed"):
                verified_completed_receipt(self.run_root, self.recipe)
            path.write_bytes(original)

    def test_operator_image_inference_accepts_only_managed_inbox_name(self):
        self._verified_run()
        config = InspectionConfig(
            self.root, self.root / "plan.json", self.root / "collection",
            self.root / "px4", profile="development",
        )
        config.workbench_inbox_root.mkdir(parents=True)
        Image.new("RGB", (16, 16), "white").save(
            config.workbench_inbox_root / "image.png"
        )
        command = build_command(config, "workbench-image-infer", parameters={
            "staged_name": "image.png", "experiment_id": self.recipe.experiment_id,
        })
        self.assertEqual(command.workflow, "visual_workbench_image_inference")
        self.assertFalse(command.requires_runtime_idle)
        self.assertIn("workbench-image-infer", command.argv)
        with self.assertRaisesRegex(ValueError, "inbox filename"):
            build_command(config, "workbench-image-infer", parameters={
                "staged_name": "../image.png",
                "experiment_id": self.recipe.experiment_id,
            })
        with self.assertRaisesRegex(ValueError, "experiment identifier"):
            build_command(config, "workbench-image-infer", parameters={
                "staged_name": "image.png", "experiment_id": "../experiment",
            })
        with self.assertRaisesRegex(ValueError, "must differ"):
            build_command(config, "workbench-image-infer", parameters={
                "staged_name": "image.png",
                "experiment_id": self.recipe.experiment_id,
                "comparison_experiment_id": self.recipe.experiment_id,
            })

    def test_local_image_inference_rejects_unverified_model_and_bad_input(self):
        source = self.root / "source.gif"
        source.write_bytes(b"GIF89a")
        with self.assertRaisesRegex(ValueError, "PNG or JPEG"):
            run_image_inference(
                self.runs, self.recipe.experiment_id, source,
                self.root / "inference/bad",
            )


if __name__ == "__main__":
    unittest.main()
