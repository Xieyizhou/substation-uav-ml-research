import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli.studies import build_parser
from src.ml.artifacts import write_json
from src.study.closed_loop_worker import _attempt_root, _probe_lidar, execute_closed_loop
from src.study.registry import ResearchRegistry
from src.study.runner import _flight_arguments


class ClosedLoopWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry_path = self.root / "registry.sqlite"
        self.registry = ResearchRegistry(self.registry_path)
        self.study_id = self.registry.create_study("candidate", "model-v1")
        matrix = [{
            "scenario_id": "simple-center-1001", "map_id": "simple",
            "target_id": "center", "seed": 1001, "condition": "ml_lidar",
        }]
        run = self.registry.ensure_runs(self.study_id, "closed-loop", matrix)[0]
        self.run = run
        self.result_path = (
            self.root / self.study_id / "closed-loop/results"
            / "simple-center-1001__ml_lidar.json"
        )
        self.queue_path = self.root / self.study_id / "closed-loop/run_queue.json"
        write_json(self.queue_path, {
            "schema_version": 1, "study_id": self.study_id, "tier": "closed-loop",
            "runs": [{
                **run, "setup_commands": [], "launcher_environment": {},
                "launcher_command": [], "flight_command": [],
                "oracle_planner_config": "planner.json",
                "result_path": str(self.result_path),
            }],
        })

    def tearDown(self):
        self.temporary.cleanup()

    def ingest(self, registry, study_id, tier, results_dir):
        if self.result_path.is_file():
            value = json.loads(self.result_path.read_text())
            if self.registry.runs(self.study_id)[0]["status"] != "completed":
                self.registry.record_metrics(self.run["run_id"], value["metrics"])
        return {
            "scheduled": 1, "imported": int(self.result_path.is_file()),
            "completed": len(self.registry.runs(
                self.study_id, statuses=("completed",)
            )),
            "pending": len(self.registry.runs(
                self.study_id, statuses=("pending", "failed", "blocked")
            )),
            "run_queue": str(self.queue_path),
        }

    def test_cli_registers_bounded_closed_loop_execution(self):
        args = build_parser().parse_args([
            "execute-closed-loop", self.study_id, "--max-runs", "1",
            "--flight-timeout", "120",
        ])
        self.assertEqual(args.command, "execute-closed-loop")
        self.assertEqual(args.max_runs, 1)
        self.assertEqual(args.flight_timeout, 120.0)

    def test_flight_queue_allows_transport_subscription_to_settle(self):
        arguments = _flight_arguments(
            "geometric_lidar", "model.onnx", "scenario.json"
        )
        timeout_index = arguments.index("--sensor-startup-timeout")
        self.assertEqual(arguments[timeout_index + 1], "20")

    def test_attempt_directories_preserve_previous_evidence(self):
        run_root = self.root / "run"
        first = _attempt_root(run_root)
        (first / "failure.json").write_text("{}")
        second = _attempt_root(run_root)
        self.assertEqual(first.name, "attempt_01")
        self.assertEqual(second.name, "attempt_02")
        self.assertTrue((first / "failure.json").is_file())

    @patch("src.study.closed_loop_worker.ensure_process_running")
    @patch("src.study.closed_loop_worker.subprocess.run")
    def test_probe_discovers_topic_without_consuming_sensor_stream(self, run, ensure):
        run.return_value.returncode = 0
        _probe_lidar(self.root / "probe.log", object(), 1.0, 5.0)
        command = run.call_args.args[0]
        self.assertEqual(command[-3:], ["sensor", "list", "--json"])
        ensure.assert_called_once()

    @patch("src.study.closed_loop_worker._run_one")
    @patch("src.study.closed_loop_worker.ingest_results")
    def test_worker_writes_result_and_completes_registry(self, ingest, run_one):
        flight_log = self.root / "flight.csv"
        flight_log.write_text("elapsed_s\n0\n")
        run_one.return_value = (flight_log, {
            "mission_success": 1, "landing_success": 1,
            "collision_count": 0, "safety_failure_count": 0,
        })
        ingest.side_effect = self.ingest
        result = execute_closed_loop(
            self.registry_path, self.study_id, self.root, max_runs=1
        )
        self.assertEqual(result["executed"], 1)
        self.assertEqual(result["completed"], 1)
        self.assertTrue(self.result_path.is_file())
        self.assertEqual(
            self.registry.runs(self.study_id)[0]["status"], "completed"
        )

    @patch("src.study.closed_loop_worker._run_one")
    @patch("src.study.closed_loop_worker.ingest_results")
    def test_worker_is_fail_fast_and_preserves_failure(self, ingest, run_one):
        ingest.side_effect = self.ingest
        run_one.side_effect = RuntimeError("simulator failed")
        with self.assertRaisesRegex(RuntimeError, "simulator failed"):
            execute_closed_loop(
                self.registry_path, self.study_id, self.root, max_runs=1
            )
        row = self.registry.runs(self.study_id)[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("simulator failed", row["failure_reason"])


if __name__ == "__main__":
    unittest.main()
