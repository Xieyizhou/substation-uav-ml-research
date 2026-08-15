import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.inspection.config import InspectionConfig
from src.ml.artifacts import object_sha256
from src.sandbox.app_smoke import run_app_smoke
from src.sandbox.beta_install import (
    inspect_beta_install_gate,
    run_beta_install_gate,
)
from src.sandbox.bootstrap import bootstrap_sandbox, inspect_bootstrap
from src.sandbox.demo_workflow import inspect_demo, run_demo
from src.sandbox.profiles import sandbox_profile
from src.sandbox.release_gate import inspect_release_gate, run_release_gate


COMMIT = "a" * 40
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SandboxReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "simulation/worlds").mkdir(parents=True)
        (self.root / "simulation/worlds/demo.sdf").write_text("<sdf/>")
        static = self.root / "src/inspection/static"
        static.mkdir(parents=True)
        for name in (
            "index.html", "app.js", "style.css", "operator.css",
            "research.css", "experiments.css", "profile.css",
        ):
            (static / name).write_text(name)
        version = {
            "sandbox_product_version": "0.1.0",
            "macos_app_version": "0.2.1",
            "operator_api_version": "1.0",
            "gate_schema_version": 2,
        }
        version_path = self.root / "config/sandbox/version.json"
        version_path.parent.mkdir(parents=True, exist_ok=True)
        version_path.write_text(json.dumps(version))
        plan = {
            "collection_plan_schema_version": 1,
            "profile": "demo",
            "scenario_count": 0,
            "scenarios": [],
        }
        plan["collection_plan_identity_sha256"] = object_sha256(plan)
        path = self.root / "config/sandbox/demo_collection_plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
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

    @patch("src.sandbox.app_smoke._read")
    @patch("src.sandbox.app_smoke.create_server")
    def test_app_smoke_checks_shell_apis_and_demo_boundary(self, server, read):
        fake = server.return_value
        fake.server_address = ("127.0.0.1", 43210)
        fake.serve_forever.return_value = None

        def response(url):
            if url.endswith("/"):
                return 200, "text/html", b"sandbox"
            if url.endswith("/api/profile"):
                return 200, "application/json", json.dumps({
                    "profile_id": "demo", "flight_enabled": False,
                }).encode()
            return 200, "application/json", b'{"ready": true}'

        read.side_effect = response
        result = run_app_smoke(self.root)
        self.assertTrue(result["passed"])
        fake.shutdown.assert_called_once()
        fake.server_close.assert_called_once()

    @patch("src.sandbox.app_smoke.create_server", side_effect=PermissionError("denied"))
    def test_app_smoke_classifies_restricted_loopback_environment(self, _server):
        result = run_app_smoke(self.root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["outcome"], "environment_unavailable")
        self.assertEqual(result["reason_code"], "loopback_bind_denied")

    @patch("src.sandbox.app_smoke._read", side_effect=OSError("closed"))
    @patch("src.sandbox.app_smoke.create_server")
    def test_app_smoke_classifies_loopback_contract_failure(self, server, _read):
        fake = server.return_value
        fake.server_address = ("127.0.0.1", 43210)
        result = run_app_smoke(self.root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["outcome"], "product_failure")
        self.assertEqual(result["reason_code"], "loopback_contract_failed")

    def test_operator_actions_use_accessible_in_page_confirmation(self):
        script = (PROJECT_ROOT / "src/inspection/static/app.js").read_text()
        markup = (PROJECT_ROOT / "src/inspection/static/index.html").read_text()
        self.assertNotIn("confirm(", script)
        self.assertNotIn("alert(", script)
        self.assertIn("requestConfirmation", script)
        self.assertIn("event.key==='Escape'", script)
        self.assertIn('id="action-dialog"', markup)
        self.assertIn('aria-live="assertive"', markup)

    @patch("src.sandbox.beta_install.git_commit", return_value=COMMIT)
    @patch("src.sandbox.beta_install._workflow_checks")
    @patch("src.sandbox.beta_install._run_check")
    @patch("src.sandbox.beta_install._tracked_files")
    def test_beta_gate_uses_clean_copy_and_is_tamper_evident(
        self, tracked, run_check, workflow, _commit,
    ):
        source = self.root / "main.py"
        source.write_text("print('demo')\n", encoding="utf-8")
        tracked.return_value = (("main.py", source),)
        run_check.return_value = {
            "name": "clean_virtual_environment", "passed": True,
            "outcome": "passed", "reason_code": None,
        }
        workflow.return_value = [
            {"name": name, "passed": True, "outcome": "passed",
             "reason_code": None} for name in (
                "bootstrap", "demo_run", "demo_inspect", "release_gate",
                "release_inspect", "app_smoke",
            )
        ]
        output = self.root / "beta"
        result = run_beta_install_gate(self.root, output)
        receipt = output / "beta_install_gate.json"
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 8)
        self.assertTrue(inspect_beta_install_gate(receipt)["passed"])
        value = json.loads(receipt.read_text())
        value["passed"] = False
        receipt.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            inspect_beta_install_gate(receipt)

    @patch("src.sandbox.beta_install.git_commit", return_value=COMMIT)
    @patch("src.sandbox.beta_install._workflow_checks")
    @patch("src.sandbox.beta_install._run_check")
    @patch("src.sandbox.beta_install._tracked_files")
    def test_beta_gate_fails_closed_when_loopback_is_unavailable(
        self, tracked, run_check, workflow, _commit,
    ):
        source = self.root / "main.py"
        source.write_text("print('demo')\n", encoding="utf-8")
        tracked.return_value = (("main.py", source),)
        run_check.return_value = {
            "name": "clean_virtual_environment", "passed": True,
            "outcome": "passed", "reason_code": None,
        }
        workflow.return_value = [
            {"name": name, "passed": True, "outcome": "passed",
             "reason_code": None} for name in (
                "bootstrap", "demo_run", "demo_inspect", "release_gate",
                "release_inspect",
            )
        ] + [{
            "name": "app_smoke", "passed": False,
            "outcome": "environment_unavailable",
            "reason_code": "loopback_bind_denied",
        }]
        output = self.root / "beta-unavailable"
        result = run_beta_install_gate(self.root, output)
        self.assertFalse(result["passed"])
        self.assertFalse(inspect_beta_install_gate(
            output / "beta_install_gate.json"
        )["passed"])


if __name__ == "__main__":
    unittest.main()
