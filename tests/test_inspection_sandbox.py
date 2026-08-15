import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.inspection.config import AccessDenied, InspectionConfig
from src.inspection.dashboard import dashboard
from src.inspection.frames import frame_page, frame_path, scenario_progress
from src.inspection.logs import log_tail
from src.inspection.runtime import LocalProcessAdapter, ProcessRecord, runtime_status
from src.inspection.service import InspectionService
from src.cli import sandbox


HASH = "a" * 64


class EmptyProcesses:
    def processes(self):
        return ()


class SecretProcess:
    def processes(self):
        return (ProcessRecord(42, 1, "/opt/px4", ("px4", "--token", "private-value")),)


class Processes:
    def __init__(self, *records):
        self.records = records

    def processes(self):
        return self.records


class InspectionFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.collection = self.root / "collection"
        self.recordings = self.collection / "recordings"
        self.recordings.mkdir(parents=True)
        (self.root / "simulation/worlds").mkdir(parents=True)
        (self.root / "simulation/worlds/empty.sdf").write_text("<sdf/>")
        self.rows = [
            self.row("dev-s", "dev-r", "development"),
            self.row("validation-s", "validation-r", "validation"),
            self.row("blind-s", "blind-r", "blind"),
        ]
        self.plan = {
            "collection_plan_identity_sha256": HASH,
            "scenarios": self.rows,
        }
        self.plan_path = self.collection / "collection_plan.json"
        self.write_json(self.plan_path, self.plan)
        self.config = InspectionConfig(
            self.root, self.plan_path, self.collection, self.root / "PX4", 0.0
        )

    def tearDown(self):
        self.temporary.cleanup()

    def row(self, scenario, recording, role):
        return {
            "scenario_id": scenario,
            "recording_id": recording,
            "dataset_role": role,
            "split": role,
            "layout_id": f"{role}-layout",
            "route_id": "inspection-route",
            "seed": 7,
            "target_class": "transformer",
        }

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_jsonl(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(value) + "\n" for value in values))

    def make_frames(self, recording_id="dev-r", count=3):
        root = self.recordings / recording_id
        frames = []
        for index in range(count):
            payload = b"\x89PNG\r\n\x1a\n" + bytes([index])
            relative = f"frames/{index:09d}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            frames.append({
                "record_type": "camera_frame",
                "frame_id": f"frame-{index}",
                "source_id": "camera",
                "sequence_number": index,
                "capture_timestamp": 10.0 + index,
                "capture_clock_domain": "simulation",
                "receive_monotonic_timestamp": 20.0 + index,
                "width": 64,
                "height": 48,
                "payload_format": "png",
                "pixel_format": "rgb8",
                "payload_relative_path": relative,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "metadata": {},
            })
        self.write_jsonl(root / "frames.jsonl", frames)
        return root, frames


class SafePathTests(InspectionFixture):
    def test_rejects_recording_and_payload_traversal(self):
        with self.assertRaises(AccessDenied):
            self.config.recording("../outside")
        with self.assertRaises(AccessDenied):
            self.config.frame_payload("dev-r", "../../outside.png")
        with self.assertRaises(AccessDenied):
            self.config.log("../outside", "recorder")

    def test_rejects_symlink_escape(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        recording = self.recordings / "dev-r"
        recording.mkdir()
        (recording / "escape.png").symlink_to(outside)
        with self.assertRaises(AccessDenied):
            self.config.frame_payload("dev-r", "escape.png")


class DashboardTests(InspectionFixture):
    def test_distinguishes_missing_recording_partial_failed_and_invalid(self):
        (self.recordings / "dev-r").mkdir()
        failed = self.recordings / "validation-r"
        self.write_json(failed / "summary.json", {})
        self.write_json(failed / "identity/recording_identity.json", {})
        self.write_json(failed / "metadata.json", {"recording_state": "in_progress"})
        invalid = self.recordings / "blind-r"
        self.write_json(invalid / "identity/collection_validation.json", {})
        view = dashboard(self.config)
        self.assertEqual(view.counts["partial"], 1)
        self.assertEqual(view.counts["failed"], 1)
        self.assertEqual(view.counts["invalid"], 1)
        self.assertEqual(view.role_counts["blind"], 1)

    def test_current_recording_and_blind_fields_are_redacted(self):
        self.write_json(self.recordings / "blind-r/live_status.json", {})
        current = dashboard(self.config).current
        self.assertEqual(current.dataset_role, "blind")
        self.assertIsNone(current.layout)
        self.assertIsNone(current.route)
        self.assertIsNone(current.seed)
        self.assertIsNone(current.target_class)


class LogAndRuntimeTests(InspectionFixture):
    def test_log_tail_is_bounded_escaped_and_classified(self):
        path = self.collection / "batch_logs/dev-s/recorder.log"
        path.parent.mkdir(parents=True)
        path.write_text("old\n<error> failure\nmission completed\n", encoding="utf-8")
        lines = log_tail(self.config, "dev-s", "recorder", 2)
        self.assertEqual([line.number for line in lines], [2, 3])
        self.assertEqual(lines[0].text, "&lt;error&gt; failure")
        self.assertEqual(lines[0].level, "error")
        self.assertEqual(lines[1].level, "success")

    def test_missing_processes_are_reported_without_commands(self):
        missing = runtime_status(EmptyProcesses())
        self.assertTrue(all(item.available for item in missing))
        self.assertTrue(all(not item.alive and item.pid is None for item in missing))
        self.assertTrue(all("no restart" in item.detail for item in missing))

    def test_process_arguments_are_not_returned(self):
        rendered = json.dumps([
            item.to_dict() for item in runtime_status(SecretProcess())
        ])
        self.assertIn('"pid": 42', rendered)
        self.assertNotIn("private-value", rendered)

    def test_shell_and_search_text_do_not_create_runtime_false_positives(self):
        adapter = Processes(
            ProcessRecord(10, 1, "/bin/zsh", (
                "zsh", "-c", "rg 'px4|gz sim|collection-run' logs",
            )),
            ProcessRecord(11, 10, "/usr/bin/rg", ("rg", "px4", "flight.log")),
            ProcessRecord(12, 10, "/usr/bin/rg", (
                "rg", "visual", "collection-run", "src",
            )),
        )
        self.assertTrue(all(not item.alive for item in runtime_status(adapter)))

    def test_executable_and_argument_tokens_identify_real_runtime(self):
        adapter = Processes(
            ProcessRecord(20, 1, "/tmp/px4", ("px4",)),
            ProcessRecord(21, 1, "/opt/gz", ("gz", "sim", "world.sdf")),
            ProcessRecord(22, 1, "/venv/python", (
                "python", "main.py", "visual", "collection-run",
            )),
            ProcessRecord(23, 1, "/venv/python", (
                "python", "main.py", "visual", "collection-record",
            )),
            ProcessRecord(24, 1, "/venv/python", (
                "python", "scripts/flight/run_task.py", "training",
            )),
        )
        items = {item.name: item for item in runtime_status(adapter)}
        self.assertTrue(all(item.alive for item in items.values()))
        self.assertEqual(items["Gazebo"].pid, 21)

    def test_process_inspection_failure_is_reported_as_unavailable(self):
        with patch("src.inspection.runtime.subprocess.run", side_effect=OSError):
            items = runtime_status(LocalProcessAdapter())
        self.assertTrue(all(not item.available for item in items))
        self.assertTrue(all("unavailable" in item.detail for item in items))

    def test_local_adapter_parses_structured_process_fields(self):
        result = type("Result", (), {
            "returncode": 0,
            "stdout": " 42  1 /opt/px4 px4 --instance 0\n",
        })()
        with patch("src.inspection.runtime.subprocess.run", return_value=result):
            records = LocalProcessAdapter().processes()
        self.assertEqual(records, (
            ProcessRecord(42, 1, "/opt/px4", ("px4", "--instance", "0")),
        ))


class FrameTests(InspectionFixture):
    def test_frame_pages_follow_manifest_without_loading_all_pngs(self):
        root, _ = self.make_frames(count=5)
        page = frame_page(self.config, "dev-r", page=2, page_size=2)
        self.assertEqual(page.total_frames, 5)
        self.assertEqual([item.sequence for item in page.frames], [2, 3])
        self.assertEqual(
            frame_path(self.config, "dev-r", "frame-3"),
            (root / "frames/000000003.png").resolve(),
        )

    def test_corrupt_optional_annotation_is_not_overlaid(self):
        root, _ = self.make_frames(count=1)
        self.write_jsonl(root / "annotations.jsonl", [{"frame_id": "frame-0"}])
        frame = frame_page(self.config, "dev-r", 1, 1).frames[0]
        self.assertEqual(frame.boxes, ())
        self.assertEqual(frame.annotation_status, "unavailable")

    def test_progress_separates_uncertain_truth_and_verified_no_target(self):
        root, frames = self.make_frames(count=2)
        self.write_jsonl(root / "synchronization.jsonl", [
            {"rgb_frame_id": "frame-0", "synchronization_status": "unmatched"},
            {"rgb_frame_id": "frame-1", "synchronization_status": "ambiguous"},
        ])
        self.write_jsonl(root / "annotations.jsonl", [self.annotation(frames[1])])
        self.write_jsonl(root / "mission_events.jsonl", [{
            "event_type": "phase_changed", "phase": "approach",
            "simulation_timestamp": 10.5, "host_command": "must-not-leak",
        }])
        progress = scenario_progress(self.config, "dev-r")
        self.assertEqual(progress.truth_counts["unmatched_or_ambiguous"], 2)
        self.assertEqual(progress.truth_counts["verified_no_target"], 1)
        self.assertEqual(progress.frame_count, 2)
        self.assertEqual(progress.elapsed_simulation_time, 1.0)
        self.assertEqual(progress.events[0]["event"], "phase_changed")
        self.assertNotIn("host_command", progress.events[0])

    def annotation(self, frame):
        return {
            "annotation_schema_version": 1, "frame_id": frame["frame_id"],
            "source_id": "camera", "sequence_number": frame["sequence_number"],
            "payload_sha256": frame["payload_sha256"], "decoded_content_sha256": HASH,
            "image_width": 64, "image_height": 48, "scenario_id": "dev-s",
            "map_id": "map", "seed": 7, "mission_phase": "other",
            "frame_order_reference": 1, "annotation_status": "verified_no_target",
            "objects": [], "recording_id": "dev-r", "dataset_identity_sha256": None,
        }


class BoundaryTests(InspectionFixture):
    def test_blind_detail_and_logs_are_not_exposed(self):
        service = InspectionService(self.config, EmptyProcesses())
        self.assertEqual(service.recordings(), [])
        with self.assertRaises(AccessDenied):
            service.frames("blind-r", 1, 10)
        with self.assertRaises(AccessDenied):
            service.logs("blind-s", "flight", 10)

    def test_queries_do_not_change_artifacts_or_mtimes(self):
        root, _ = self.make_frames(count=2)
        log = self.collection / "batch_logs/dev-s/flight.log"
        log.parent.mkdir(parents=True)
        log.write_text("completed\n", encoding="utf-8")
        paths = (self.plan_path, root / "frames.jsonl", root / "frames/000000000.png", log)
        before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths}
        service = InspectionService(self.config, EmptyProcesses())
        service.dashboard(); service.runtime(); service.logs("dev-s", "flight", 10)
        service.frames("dev-r", 1, 1); service.progress("dev-r"); service.storage()
        after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths}
        self.assertEqual(before, after)

    def test_version_manifest_is_exposed_as_a_read_only_record(self):
        version = {
            "sandbox_product_version": "0.1.0",
            "macos_app_version": "0.2.1",
            "operator_api_version": "1.0",
            "gate_schema_version": 2,
        }
        self.write_json(self.root / "config/sandbox/version.json", version)
        self.assertEqual(
            InspectionService(self.config, EmptyProcesses()).version(), version
        )


class SandboxCliTests(InspectionFixture):
    def test_doctor_and_status_are_read_only(self):
        self.make_frames(count=1)
        with patch.object(
            sandbox.InspectionConfig, "defaults", return_value=self.config
        ), patch("builtins.print"):
            self.assertEqual(sandbox.main(["doctor"]), 0)
            self.assertEqual(sandbox.main(["status"]), 0)

    def test_help_exposes_local_inspection_workflow(self):
        help_text = sandbox.build_parser().format_help()
        for command in (
            "bootstrap", "doctor", "status", "storage", "retention-plan",
            "serve", "demo-run", "release-gate", "beta-install-gate",
            "development-app-gate",
        ):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
