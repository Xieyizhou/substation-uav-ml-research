import json
from pathlib import Path
import tempfile
import unittest

from src.inspection.config import InspectionConfig
from src.inspection.experiments import experiment_summaries
from src.ml.artifacts import object_sha256, write_json
from tests.test_sandbox_experiment_runner import recipe


class SandboxExperimentInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = InspectionConfig(
            self.root, self.root / "plan.json", self.root / "collection",
            self.root / "PX4", 0.0,
        )
        self.directory = self.config.sandbox_experiments_root / "validation-416-every2"
        self.directory.mkdir(parents=True)
        self.recipe = recipe()
        write_json(self.directory / "recipe.json", self.recipe.to_record())

    def tearDown(self):
        self.temporary.cleanup()

    def test_ready_recipe_is_visible_without_raw_data(self):
        write_json(self.directory / "status.json", {
            "sandbox_experiment_status_schema_version": 1,
            "state": "ready",
            "recipe_identity_sha256": self.recipe.recipe_identity_sha256,
        })
        rows = experiment_summaries(self.config)
        self.assertEqual(rows[0]["state"], "ready")
        self.assertEqual(rows[0]["partition"], "validation")
        self.assertNotIn("raw_predictions", json.dumps(rows))

    def test_status_identity_mismatch_is_reported_as_invalid(self):
        write_json(self.directory / "status.json", {
            "sandbox_experiment_status_schema_version": 1,
            "state": "complete",
            "recipe_identity_sha256": "f" * 64,
        })
        rows = experiment_summaries(self.config)
        self.assertEqual(rows[0]["state"], "invalid")
        self.assertIn("different recipe", rows[0]["error"])

    def test_complete_summary_exposes_metrics_without_predictions(self):
        result = {
            "recipe_identity_sha256": self.recipe.recipe_identity_sha256,
            "metrics": {
                "precision": 0.9, "recall": 0.8,
                "small_object_recall": 0.7,
                "no_target_false_positive_rate": 0.01,
            },
            "timing": {"end_to_end_ms": {"p95_ms": 12.0}},
            "resources": {"throughput_fps": 40.0},
        }
        result["result_identity_sha256"] = object_sha256(result)
        write_json(self.directory / "result.json", result)
        write_json(self.directory / "status.json", {
            "sandbox_experiment_status_schema_version": 1,
            "state": "complete",
            "recipe_identity_sha256": self.recipe.recipe_identity_sha256,
            "result_identity_sha256": result["result_identity_sha256"],
        })
        row = experiment_summaries(self.config)[0]
        self.assertEqual(row["metrics"]["precision"], 0.9)
        self.assertNotIn("predictions", json.dumps(row))


if __name__ == "__main__":
    unittest.main()
