from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.cli import visual
from src.ml.visual_collection import build_collection_plan
from src.ml.visual_collection_batch import (
    archive_failed_attempt,
    run_collection_batch,
)
from src.ml.visual_collection_process import (
    CollectionProcessError,
    ensure_process_running,
    wait_for_flight,
)


class VisualCollectionBatchTests(unittest.TestCase):
    def test_recorder_failure_interrupts_flight_wait(self):
        flight = Mock()
        flight.name = "flight task"
        flight.log_path = Path("flight.log")
        flight.process.poll.return_value = None
        recorder = Mock()
        recorder.name = "visual recorder"
        recorder.log_path = Path("recorder.log")
        recorder.process.poll.return_value = 1
        with self.assertRaisesRegex(
            CollectionProcessError,
            "visual recorder exited with code 1",
        ):
            wait_for_flight(flight, recorder, 60.0)
        recorder.close_log.assert_called_once_with()

    def test_failed_launcher_is_rejected_before_topic_probe(self):
        managed = Mock()
        managed.name = "PX4/Gazebo launcher"
        managed.log_path = Path("simulator.log")
        managed.process.poll.return_value = 1
        with patch("src.ml.visual_collection_process.time.sleep"):
            with self.assertRaisesRegex(
                CollectionProcessError,
                "exited with code 1",
            ):
                ensure_process_running(managed)
        managed.close_log.assert_called_once_with()

    def test_cli_exposes_resumable_batch_command(self):
        parser = visual.build_parser()
        args = parser.parse_args(
            [
                "collection-run",
                "--plan",
                "plan.json",
                "--dry-run",
                "--max-scenarios",
                "2",
            ]
        )
        self.assertEqual(args.command, "collection-run")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.max_scenarios, 2)
        self.assertEqual(args.max_attempts, 3)
        self.assertEqual(args.recorder_timeout, 900.0)

    def test_failed_attempt_is_archived_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recordings" / "recording-1"
            logs = root / "batch_logs" / "scenario-1"
            recording.mkdir(parents=True)
            logs.mkdir(parents=True)
            (recording / "metadata.json").write_text("{}", encoding="utf-8")
            (logs / "flight.log").write_text("failed", encoding="utf-8")

            destination = archive_failed_attempt(
                root,
                "scenario-1",
                "recording-1",
            )

            self.assertEqual(destination.name, "attempt_01")
            self.assertTrue((destination / "recording" / "metadata.json").exists())
            self.assertTrue((destination / "logs" / "flight.log").exists())
            self.assertFalse(recording.exists())
            self.assertFalse(logs.exists())

    def test_dry_run_lists_missing_scenarios_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_collection_plan()
            result = run_collection_batch(
                plan,
                root / "collection_plan.json",
                root,
                max_scenarios=2,
                dry_run=True,
            )
            self.assertEqual(result["processed_count"], 2)
            self.assertEqual(
                [row["scenario_id"] for row in result["processed"]],
                ["training-top_right-2001", "training-center-2002"],
            )
            self.assertFalse((root / "scenarios").exists())


if __name__ == "__main__":
    unittest.main()
