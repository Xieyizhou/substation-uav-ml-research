import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from src.inspection.config import InspectionConfig
from src.inspection.app import create_server
from src.sandbox.job_commands import SandboxCommand, build_command
from src.sandbox.job_models import SandboxJob, SandboxJobStore, utc_now
from src.sandbox.operator import OperatorBusy, SandboxOperator


HASH = "a" * 64


class EmptyProcesses:
    def processes(self):
        return ()


class FakeOperator:
    def __init__(self):
        self.started = []
        self.closed = False

    def status(self):
        return {"state": "idle", "active_job": None, "history": []}

    def start(self, action, scenario_id=None, parameters=None):
        self.started.append((action, scenario_id, parameters))
        return {"job_id": "job-1", "state": "preparing"}

    def stop(self, job_id):
        return {"job_id": job_id, "state": "stopping"}

    def log(self, job_id, limit):
        return [f"{job_id}:{limit}"]

    def shutdown(self):
        self.closed = True


class SandboxOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.collection = self.root / "collection"
        self.collection.mkdir()
        self.plan_path = self.collection / "collection_plan.json"
        self.plan = {
            "collection_plan_identity_sha256": HASH,
            "scenarios": [
                self.row("development-s", "development-r", "development"),
                self.row("blind-s", "blind-r", "blind"),
            ],
        }
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.config = InspectionConfig(
            self.root,
            self.plan_path,
            self.collection,
            self.root / "PX4",
            0.0,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def row(self, scenario_id, recording_id, split):
        return {
            "scenario_id": scenario_id,
            "recording_id": recording_id,
            "dataset_role": "held_out_test" if split == "blind" else split,
            "split": split,
            "map_id": split,
            "target_id": "transformer",
            "route_id": "route",
            "seed": 1,
        }

    def wait_idle(self, operator, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = operator.status()
            if status["state"] == "idle":
                return status
            time.sleep(0.02)
        self.fail("sandbox operator did not become idle")

    def test_command_allowlist_rejects_unknown_and_blind_smoke(self):
        with self.assertRaisesRegex(ValueError, "unsupported sandbox action"):
            build_command(self.config, "shell")
        with self.assertRaisesRegex(ValueError, "blind scenario"):
            build_command(self.config, "flight-smoke", "blind-s")
        smoke = build_command(self.config, "flight-smoke", "development-s")
        self.assertNotIn("blind-s", smoke.argv)
        view = build_command(self.config, "training-view-v2")
        self.assertIn("training-view-materialize", view.argv)
        with self.assertRaisesRegex(ValueError, "materialize the v2 training view"):
            build_command(self.config, "training-smoke-v2")

    def test_training_smoke_uses_a_new_managed_output_directory(self):
        identity = self.root / (
            "data/research/visual_yolo_v2/identity/training_view_identity.json"
        )
        identity.parent.mkdir(parents=True)
        identity.write_text("{}", encoding="utf-8")
        command = build_command(self.config, "training-smoke-v2")
        output = command.argv[command.argv.index("--output") + 1]
        self.assertTrue(output.startswith("outputs/sandbox/training_smoke/"))

    def test_package_inspection_is_fixed_and_requires_frozen_manifest(self):
        with self.assertRaisesRegex(ValueError, "package is not present"):
            build_command(self.config, "package-inspect-v2")
        package = self.root / "models/equipment/visual-yolo11n-baseline-v2-package"
        package.mkdir(parents=True)
        (package / "manifest.json").write_text("{}", encoding="utf-8")
        command = build_command(self.config, "package-inspect-v2")
        self.assertIn("model-package-inspect", command.argv)
        self.assertEqual(command.argv[-1], str(package))
        self.assertEqual(command.timeout_s, 120.0)

    @patch("src.sandbox.job_commands.materialize_recipe")
    def test_experiment_command_uses_validated_recipe_and_fixed_cli(self, materialize):
        recipe = Mock(source_frame_count=64)
        output = self.root / "outputs/sandbox/experiments/validation-416"
        materialize.return_value = (recipe, output)
        parameters = {
            "name": "validation-416", "partition": "validation",
            "input_size": 416, "frame_skip_interval": 1, "frame_limit": 64,
        }
        command = build_command(
            self.config, "experiment-run", parameters=parameters
        )
        self.assertEqual(command.action, "experiment-run")
        self.assertEqual(command.timeout_s, 900.0)
        self.assertEqual(command.argv[-1], str(output / "recipe.json"))
        materialize.assert_called_once()
        with self.assertRaisesRegex(ValueError, "unsupported experiment parameter"):
            build_command(
                self.config, "experiment-run",
                parameters={**parameters, "model_path": "/tmp/model.onnx"},
            )

    def test_job_store_detects_record_tampering(self):
        store = SandboxJobStore(self.config.sandbox_jobs_root)
        job = SandboxJob("job-1", "doctor", "complete", utc_now(), 5.0)
        store.write(job)
        path = store.directory(job.job_id) / "job.json"
        record = json.loads(path.read_text())
        record["action"] = "changed"
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            store.read(job.job_id)

    def test_managed_job_completes_and_persists_bounded_log(self):
        command = SandboxCommand(
            "doctor",
            (sys.executable, "-c", "print('sandbox-ready')"),
            5.0,
        )
        operator = SandboxOperator(self.config, EmptyProcesses())
        with patch("src.sandbox.operator.build_command", return_value=command):
            started = operator.start("doctor")
            self.assertEqual(started["state"], "preparing")
            status = self.wait_idle(operator)
        self.assertEqual(status["history"][0]["state"], "complete")
        self.assertIn("sandbox-ready", "\n".join(operator.log(started["job_id"])))

    def test_single_instance_lock_and_safe_stop(self):
        command = SandboxCommand(
            "doctor",
            (sys.executable, "-c", "import time; time.sleep(30)"),
            60.0,
        )
        first = SandboxOperator(self.config, EmptyProcesses())
        second = SandboxOperator(self.config, EmptyProcesses())
        with patch("src.sandbox.operator.build_command", return_value=command):
            started = first.start("doctor")
            deadline = time.monotonic() + 2.0
            while first.status()["state"] == "preparing" and time.monotonic() < deadline:
                time.sleep(0.01)
            with self.assertRaisesRegex(OperatorBusy, "another sandbox operator"):
                second.start("doctor")
            first.stop(started["job_id"])
            status = self.wait_idle(first)
        self.assertEqual(status["history"][0]["state"], "failed")
        self.assertTrue(status["history"][0]["stop_requested"])

    def test_live_interrupted_job_blocks_a_new_operator_job(self):
        store = SandboxJobStore(self.config.sandbox_jobs_root)
        interrupted = SandboxJob(
            "job-old", "doctor", "running", utc_now(), 60.0, pid=43210
        )
        store.write(interrupted)
        with patch("src.sandbox.operator.process_alive", return_value=True):
            operator = SandboxOperator(self.config, EmptyProcesses())
            with self.assertRaisesRegex(OperatorBusy, "interrupted sandbox job"):
                operator.start("doctor")

    def test_http_mutations_require_server_token(self):
        fake = FakeOperator()
        try:
            server = create_server(
                self.config,
                host="127.0.0.1",
                port=0,
                adapter=EmptyProcesses(),
                operator=fake,
            )
        except PermissionError:
            self.skipTest("local socket binding is unavailable")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(*server.server_address)
        try:
            connection.request("GET", "/api/operator")
            response = connection.getresponse()
            status = json.loads(response.read())
            token = status["operator_token"]
            connection.request("GET", "/api/research")
            research = connection.getresponse()
            payload = json.loads(research.read())
            self.assertEqual(research.status, 200)
            self.assertEqual(payload["stage_count"], 4)
            connection.request("GET", "/api/experiments")
            experiments = connection.getresponse()
            self.assertEqual(experiments.status, 200)
            self.assertEqual(json.loads(experiments.read()), [])
            connection.request(
                "POST",
                "/api/operator/start",
                json.dumps({"action": "doctor"}),
                {"Content-Type": "application/json"},
            )
            self.assertEqual(connection.getresponse().status, 403)
            connection.request(
                "POST",
                "/api/operator/start",
                json.dumps({"action": "doctor"}),
                {
                    "Content-Type": "application/json",
                    "X-Sandbox-Token": token,
                },
            )
            accepted = connection.getresponse()
            accepted.read()
            self.assertEqual(accepted.status, 202)
            self.assertEqual(fake.started, [("doctor", None, None)])
            parameters = {
                "name": "validation-416-test", "partition": "validation",
                "input_size": 416, "frame_skip_interval": 1, "frame_limit": 64,
            }
            connection.request(
                "POST", "/api/operator/start",
                json.dumps({"action": "experiment-run", "parameters": parameters}),
                {"Content-Type": "application/json", "X-Sandbox-Token": token},
            )
            experiment = connection.getresponse()
            experiment.read()
            self.assertEqual(experiment.status, 202)
            self.assertEqual(fake.started[-1], ("experiment-run", None, parameters))
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
        self.assertTrue(fake.closed)

    def test_server_rejects_non_loopback_binding(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            create_server(
                self.config,
                host="0.0.0.0",
                port=0,
                adapter=EmptyProcesses(),
                operator=FakeOperator(),
            )


if __name__ == "__main__":
    unittest.main()
