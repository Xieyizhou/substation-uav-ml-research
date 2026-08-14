import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli.studies import build_parser
from src.ml.artifacts import object_sha256, write_json
from src.study.closed_loop_worker import (
    _attempt_root,
    _is_retryable_startup_failure,
    _probe_lidar,
    _run_one,
    _verify_replay_receipt,
    execute_closed_loop,
    execute_formal,
)
from src.study.flight_budget import (
    closed_loop_timeout_s,
    flight_timeout_policy,
    route_length_m,
)
from src.study.registry import ResearchRegistry
from src.study.runner import _flight_arguments
from src.vision.collection.process import CollectionProcessError


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
        planner_path = self.write_planner()
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
                "oracle_planner_config": str(planner_path),
                "result_path": str(self.result_path),
            }],
        })

    def write_planner(self, goal=(60, 0), resolution=1.0):
        path = self.root / "planner.json"
        write_json(path, {
            "width": 80, "height": 2, "start_cell": [0, 0],
            "goal_cell": list(goal), "obstacles": [],
            "resolution_m": resolution,
        })
        return path

    def tearDown(self):
        self.temporary.cleanup()

    def eligible_formal_gate(self):
        registry = ResearchRegistry(self.root / "formal.sqlite")
        manifest = {
            "model_id": "formal-model",
            "onnx_sha256": "a" * 64,
            "dataset_id": "formal-dataset",
            "dataset_sha256": "b" * 64,
            "parent_model": None,
        }
        registry.register_model(manifest, self.root / "formal-model")
        study_id = registry.create_study("formal", "formal-model")
        matrix = [
            {
                "scenario_id": f"simple-center-{1001 + index}",
                "map_id": "simple", "target_id": "center",
                "seed": 1001 + index, "condition": condition,
            }
            for condition in (
                "geometric_lidar", "ml_lidar", "geometric_ml_fusion"
            )
            for index in range(5)
        ]
        for run in registry.ensure_runs(study_id, "closed-loop", matrix):
            registry.record_metrics(run["run_id"], {
                "mission_success": 1, "landing_success": 1,
                "collision_count": 0, "safety_failure_count": 0,
                "sensor_healthy_ratio": 1.0,
                "predicted_danger_sample_count": 1,
                "replan_attempt_count": 1,
                "successful_replan_count": 1,
                "active_replan_count": 1,
            })
        receipt = {
            "schema_version": 1,
            "passed": True,
            "model_id": "formal-model",
            "model_sha256": "a" * 64,
            "dataset_id": "formal-dataset",
        }
        receipt["replay_gate_identity_sha256"] = object_sha256(receipt)
        path = self.root / "replay_gate.json"
        write_json(path, receipt)
        return registry, study_id, path

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
            "--flight-timeout", "120", "--scenario-id", "simple-center-1001",
        ])
        self.assertEqual(args.command, "execute-closed-loop")
        self.assertEqual(args.max_runs, 1)
        self.assertEqual(args.flight_timeout, 120.0)
        self.assertEqual(args.scenario_id, "simple-center-1001")

    def test_cli_exposes_challenge_execution_and_tier_aware_capabilities(self):
        challenge = build_parser().parse_args([
            "execute-challenge", self.study_id, "--max-runs", "1",
        ])
        capabilities = build_parser().parse_args([
            "capabilities", self.study_id, "--tier", "formal",
        ])
        self.assertEqual(challenge.command, "execute-challenge")
        self.assertEqual(challenge.max_runs, 1)
        self.assertEqual(capabilities.tier, "formal")

    def test_cli_requires_explicit_replay_receipt_for_formal_execution(self):
        args = build_parser().parse_args([
            "execute-formal", self.study_id, "--replay-gate", "gate.json",
            "--qualification-study", "qualification-study",
            "--max-runs", "1",
        ])
        self.assertEqual(args.command, "execute-formal")
        self.assertEqual(args.replay_gate, Path("gate.json"))
        self.assertEqual(args.qualification_study, "qualification-study")

    def test_formal_worker_requires_current_capability_receipt(self):
        registry, study_id, replay = self.eligible_formal_gate()
        with self.assertRaisesRegex(ValueError, "capability challenge receipt"):
            execute_formal(
                registry.path, study_id, self.root, replay_gate_path=replay,
                max_runs=1,
            )

    def test_flight_queue_allows_transport_subscription_to_settle(self):
        arguments = _flight_arguments(
            "geometric_lidar", "model.onnx", "scenario.json"
        )
        timeout_index = arguments.index("--sensor-startup-timeout")
        self.assertEqual(arguments[timeout_index + 1], "20")
        stale_index = arguments.index("--sensor-stale-after")
        self.assertEqual(arguments[stale_index + 1], "2.0")

    def test_route_aware_timeout_covers_long_round_trip(self):
        planner = self.write_planner()
        self.assertEqual(route_length_m(planner), 60.0)
        self.assertGreater(closed_loop_timeout_s(planner), 600.0)
        self.assertLessEqual(closed_loop_timeout_s(planner), 900.0)

    def test_route_aware_timeout_covers_waypoint_and_landing_overhead(self):
        planner = self.write_planner(goal=(69, 0))
        self.assertGreater(closed_loop_timeout_s(planner), 800.0)
        self.assertEqual(flight_timeout_policy()["max_timeout_s"], 900.0)

    def test_route_aware_timeout_is_bounded_and_override_is_exact(self):
        planner = self.write_planner(goal=(1, 0), resolution=0.5)
        self.assertGreaterEqual(closed_loop_timeout_s(planner), 240.0)
        self.assertEqual(closed_loop_timeout_s(planner, 123.0), 123.0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            closed_loop_timeout_s(planner, 0.0)

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
        run.return_value.stdout = '{"lidar_2d": "/world/test/scan"}\n'
        topic = _probe_lidar(self.root / "probe.log", object(), 1.0, 5.0)
        command = run.call_args.args[0]
        self.assertEqual(command[-3:], ["sensor", "list", "--json"])
        self.assertEqual(topic, "/world/test/scan")
        ensure.assert_called_once()

    def test_preflight_transport_failures_without_telemetry_are_retryable(self):
        attempt = self.root / "attempt"
        attempt.mkdir()
        (attempt / "flight.log").write_text(
            "MAVSDK connection failed after 2 attempts\n", encoding="utf-8"
        )
        self.assertTrue(_is_retryable_startup_failure(
            CollectionProcessError("expected one new flight log, found 0"),
            attempt,
        ))
        self.assertFalse(_is_retryable_startup_failure(
            CollectionProcessError("flight task exceeded timeout"), attempt
        ))
        (attempt / "flight.log").write_text(
            "sensor gazebo_lidar_2d did not become ready within 20s: "
            "waiting for first scan\n",
            encoding="utf-8",
        )
        self.assertTrue(_is_retryable_startup_failure(
            CollectionProcessError("expected one new flight log, found 0"),
            attempt,
        ))

    @patch("src.study.closed_loop_worker.mission_metrics", return_value={})
    @patch("src.study.closed_loop_worker.landed_mission_status")
    @patch("src.study.closed_loop_worker._new_flight_log")
    @patch("src.study.closed_loop_worker.wait_for_study_flight")
    @patch("src.study.closed_loop_worker.stop_process")
    @patch("src.study.closed_loop_worker.ensure_process_running")
    @patch("src.study.closed_loop_worker.start_process")
    @patch("src.study.closed_loop_worker._probe_lidar", return_value="/world/test/scan")
    @patch("src.study.closed_loop_worker._run_setup")
    def test_run_uses_discovered_runtime_topic(
        self, setup, probe, start, ensure, stop, wait, new_log, status, metrics
    ):
        new_log.return_value = self.root / "flight.csv"
        status.return_value = {
            "status": "completed", "landing_confirmed": True,
        }
        start.side_effect = [object(), object()]
        row = {
            "setup_commands": [], "launcher_environment": {},
            "launcher_command": ["launcher"],
            "flight_command": ["python", "main.py", "task"],
            "oracle_planner_config": "planner.json",
        }
        _run_one(row, self.root / "run", startup_timeout_s=1.0,
                 probe_timeout_s=1.0, flight_timeout_s=1.0)
        flight_command = start.call_args_list[1].args[1]
        self.assertEqual(flight_command[-2:], ["--sensor-topic", "/world/test/scan"])
        self.assertTrue(start.call_args_list[0].kwargs["discard_stdout"])

    @patch("src.study.closed_loop_worker._run_one")
    @patch("src.study.closed_loop_worker.ingest_results")
    def test_worker_writes_result_and_completes_registry(self, ingest, run_one):
        flight_log = self.root / "flight.csv"
        flight_log.write_text("elapsed_s\n0\n")
        run_one.return_value = (
            flight_log,
            {
                "mission_success": 1, "landing_success": 1,
                "collision_count": 0, "safety_failure_count": 0,
            },
            {"status": "completed", "landing_confirmed": True},
        )
        ingest.side_effect = self.ingest
        result = execute_closed_loop(
            self.registry_path, self.study_id, self.root, max_runs=1
        )
        self.assertEqual(result["executed"], 1)
        self.assertEqual(result["completed"], 1)
        self.assertTrue(run_one.call_args.kwargs["allow_progress_extension"])
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

    @patch("src.study.closed_loop_worker.mission_metrics")
    @patch("src.study.closed_loop_worker.landed_mission_status")
    @patch("src.study.closed_loop_worker._new_flight_log")
    @patch("src.study.closed_loop_worker.wait_for_study_flight")
    @patch("src.study.closed_loop_worker.stop_process")
    @patch("src.study.closed_loop_worker.ensure_process_running")
    @patch("src.study.closed_loop_worker.start_process")
    @patch("src.study.closed_loop_worker._probe_lidar", return_value="/scan")
    @patch("src.study.closed_loop_worker._run_setup")
    def test_run_accepts_nonzero_process_with_classified_mission_failure(
        self, setup, probe, start, ensure, stop, wait, new_log, status, metrics
    ):
        start.side_effect = [object(), object()]
        wait.side_effect = CollectionProcessError("flight exited with code 1")
        new_log.return_value = self.root / "failed.csv"
        status.return_value = {
            "status": "failed", "landing_confirmed": True,
            "message": "TimeoutError: Timed out before reaching WP05",
        }
        metrics.return_value = {"mission_success": 0, "landing_success": 1}
        row = {
            "setup_commands": [], "launcher_environment": {},
            "launcher_command": ["launcher"],
            "flight_command": ["python", "main.py", "task"],
            "oracle_planner_config": "planner.json",
        }
        _, result, mission = _run_one(
            row, self.root / "run-failed", startup_timeout_s=1.0,
            probe_timeout_s=1.0, flight_timeout_s=1.0,
        )
        self.assertEqual(result["mission_success"], 0)
        self.assertEqual(mission["status"], "failed")

    @patch("src.study.closed_loop_worker._run_one")
    @patch("src.study.closed_loop_worker.ingest_results")
    def test_worker_ingests_classified_mission_failure_and_continues(
        self, ingest, run_one
    ):
        flight_log = self.root / "failed-flight.csv"
        flight_log.write_text("elapsed_s\n0\n")
        run_one.return_value = (
            flight_log,
            {"mission_success": 0, "landing_success": 1},
            {
                "status": "failed", "landing_confirmed": True,
                "message": "TimeoutError: Timed out before reaching WP05",
            },
        )
        ingest.side_effect = self.ingest
        result = execute_closed_loop(
            self.registry_path, self.study_id, self.root, max_runs=1
        )
        self.assertEqual(result["completed"], 1)
        payload = json.loads(self.result_path.read_text())
        self.assertEqual(payload["mission"]["outcome_class"], "mission_failure")
        self.assertEqual(payload["metrics"]["mission_success"], 0)

    @patch("src.study.closed_loop_worker._run_one")
    @patch("src.study.closed_loop_worker.ingest_results")
    def test_worker_restarts_simulator_after_preflight_mavsdk_failure(
        self, ingest, run_one
    ):
        flight_log = self.root / "flight.csv"
        flight_log.write_text("elapsed_s\n0\n")

        def run_side_effect(row, attempt, **options):
            if run_one.call_count == 1:
                (attempt / "flight.log").write_text(
                    "MAVSDK connection failed after 2 attempts\n", encoding="utf-8"
                )
                raise CollectionProcessError(
                    "expected one new flight log, found 0"
                )
            return flight_log, {}, {
                "status": "completed", "landing_confirmed": True,
            }

        run_one.side_effect = run_side_effect
        ingest.side_effect = self.ingest
        result = execute_closed_loop(
            self.registry_path, self.study_id, self.root, max_runs=1
        )
        self.assertEqual(result["completed"], 1)
        self.assertEqual(run_one.call_count, 2)
        attempts = self.root / self.study_id / "closed-loop/runs" / self.run["run_id"] / "attempts"
        self.assertTrue((attempts / "attempt_01/failure.json").is_file())
        self.assertTrue((attempts / "attempt_02").is_dir())

    def test_formal_worker_rejects_missing_replay_receipt(self):
        with self.assertRaisesRegex(ValueError, "requires --replay-gate"):
            execute_formal(
                self.registry_path, self.study_id, self.root, max_runs=1
            )

    def test_formal_gate_binds_replay_to_candidate_and_closed_loop(self):
        registry, study_id, path = self.eligible_formal_gate()
        receipt = _verify_replay_receipt(registry, study_id, path)
        self.assertTrue(receipt["passed"])

    def test_formal_gate_can_reuse_a_same_candidate_qualification_study(self):
        registry, qualification_id, path = self.eligible_formal_gate()
        formal_id = registry.create_study("formal-evidence", "formal-model")
        receipt = _verify_replay_receipt(
            registry, formal_id, path, qualification_id
        )
        self.assertTrue(receipt["passed"])

    def test_formal_gate_rejects_tampered_receipt(self):
        registry, study_id, path = self.eligible_formal_gate()
        receipt = json.loads(path.read_text())
        receipt["model_sha256"] = "c" * 64
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            _verify_replay_receipt(registry, study_id, path)

    def test_formal_gate_rejects_identity_valid_wrong_model(self):
        registry, study_id, path = self.eligible_formal_gate()
        receipt = json.loads(path.read_text())
        receipt.pop("replay_gate_identity_sha256")
        receipt["model_sha256"] = "c" * 64
        receipt["replay_gate_identity_sha256"] = object_sha256(receipt)
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "ONNX hash"):
            _verify_replay_receipt(registry, study_id, path)


if __name__ == "__main__":
    unittest.main()
