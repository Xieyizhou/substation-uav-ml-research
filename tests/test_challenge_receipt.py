import json
from pathlib import Path
import tempfile
import unittest

from src.ml.artifacts import write_json
from src.study.challenge_receipt import (
    inspect_challenge_receipt,
    materialize_challenge_receipt,
)
from src.study.challenge_spec import challenge_matrix
from src.study.registry import ResearchRegistry


class ChallengeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry_path = self.root / "registry.sqlite"
        self.results = self.root / "results"
        registry = ResearchRegistry(self.registry_path)
        registry.register_model(
            {
                "model_id": "model-1", "onnx_sha256": "a" * 64,
                "dataset_id": "dataset-1",
            },
            self.root / "model",
        )
        self.study_id = registry.create_study("challenge", "model-1")
        runs = registry.ensure_runs(
            self.study_id, "challenge", challenge_matrix(),
            config_hash="config-1", model_hash="a" * 64,
        )
        for run in runs:
            metrics = {
                "mission_success": 1, "landing_success": 1,
                "collision_count": 0, "sensor_healthy_ratio": 0.99,
                "predicted_danger_sample_count": 1,
                "replan_attempt_count": 1, "successful_replan_count": 1,
                "active_replan_count": 1,
            }
            registry.record_metrics(run["run_id"], metrics)
            output = self.results / self.study_id / "challenge" / "results" / (
                f"{run['scenario_id']}__{run['condition']}.json"
            )
            write_json(output, {
                "run_id": run["run_id"], "scenario_id": run["scenario_id"],
                "condition": run["condition"], "code_commit": "commit-1",
                "metrics": metrics,
            })

    def tearDown(self):
        self.temporary.cleanup()

    def test_materializes_and_detects_result_tampering(self):
        receipt = materialize_challenge_receipt(
            self.registry_path, self.study_id, self.results
        )
        self.assertTrue(receipt["passed"])
        inspected = inspect_challenge_receipt(
            receipt["path"], project_root=self.root,
            registry_path=self.registry_path,
        )
        self.assertFalse(inspected["current"])
        result = next(
            (Path(receipt["path"]).parent / "results").glob("*.json")
        )
        value = json.loads(result.read_text(encoding="utf-8"))
        value["metrics"]["collision_count"] = 1
        result.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "result changed"):
            inspect_challenge_receipt(receipt["path"])


if __name__ == "__main__":
    unittest.main()
