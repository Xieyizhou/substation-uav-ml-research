import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.inspection.config import InspectionConfig
from src.ml.artifacts import object_sha256
from src.sandbox.bootstrap import bootstrap_sandbox, inspect_bootstrap
from src.sandbox.demo_workflow import inspect_demo, run_demo
from src.sandbox.profiles import sandbox_profile
from src.sandbox.release_gate import inspect_release_gate, run_release_gate


COMMIT = "a" * 40


class SandboxReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "simulation/worlds").mkdir(parents=True)
        (self.root / "simulation/worlds/demo.sdf").write_text("<sdf/>")
        static = self.root / "src/inspection/static"
        static.mkdir(parents=True)
        for name in ("index.html", "app.js", "style.css", "profile.css"):
            (static / name).write_text(name)
        plan = {
            "collection_plan_schema_version": 1,
            "profile": "demo",
            "scenario_count": 0,
            "scenarios": [],
        }
        plan["collection_plan_identity_sha256"] = object_sha256(plan)
        path = self.root / "config/sandbox/demo_collection_plan.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(plan))
        self.config = InspectionConfig.for_profile(self.root, "demo")

    def tearDown(self):
        self.temporary.cleanup()

    def test_profile_keeps_demo_separate_from_full_runtime(self):
        demo = sandbox_profile("demo")
        self.assertFalse(demo.flight_enabled)
        self.assertEqual(demo.available_workflows, ("demo_contract",))
        self.assertIn("outputs/sandbox/demo", str(self.config.collection_root))
        with self.assertRaisesRegex(ValueError, "unsupported sandbox profile"):
            sandbox_profile("unknown")

    @patch("src.sandbox.bootstrap.git_commit", return_value=COMMIT)
    def test_bootstrap_is_idempotent_and_detects_tampering(self, _commit):
        path = self.root / "receipt/bootstrap.json"
        first = bootstrap_sandbox(self.config, path)
        second = bootstrap_sandbox(self.config, path)
        self.assertEqual(first["bootstrap_identity_sha256"],
                         second["bootstrap_identity_sha256"])
        self.assertTrue(inspect_bootstrap(path)["ready"])
        value = json.loads(path.read_text())
        value["ready"] = False
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            inspect_bootstrap(path)

    @patch("src.sandbox.demo_workflow.git_commit", return_value=COMMIT)
    def test_demo_is_deterministic_non_formal_and_tamper_evident(self, _commit):
        first = self.root / "demo/first"
        second = self.root / "demo/second"
        run_demo(self.root, first)
        run_demo(self.root, second)
        inspected = inspect_demo(first)
        self.assertTrue(inspected["passed"])
        self.assertEqual(inspected["metrics"]["macro_f1"], 1.0)
        self.assertEqual(
            (first / "demo_result.json").read_bytes(),
            (second / "demo_result.json").read_bytes(),
        )
        value = json.loads((first / "demo_result.json").read_text())
        value["passed"] = False
        (first / "demo_result.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            inspect_demo(first)

    @patch("src.sandbox.release_gate.git_commit", return_value=COMMIT)
    def test_release_gate_passes_and_is_inspectable(self, _commit):
        output = self.root / "release"
        result = run_release_gate(self.config, output)
        self.assertTrue(result["passed"])
        self.assertEqual(set(result["checks"]), {
            "bootstrap", "demo_workflow", "doctor", "web_assets",
        })
        self.assertTrue(inspect_release_gate(output / "release_gate.json")["passed"])


if __name__ == "__main__":
    unittest.main()
