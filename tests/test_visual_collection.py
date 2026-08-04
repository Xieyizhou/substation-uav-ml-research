import json
from pathlib import Path
import tempfile
import unittest

from tests.test_gazebo_visual_pilot import (
    box,
    image_message,
    truth_message,
)
from src.flight.mission_events import MissionEventWriter
from src.ml.artifacts import object_sha256
from src.vision.collection.gazebo_truth import parse_gazebo_truth_message
from src.vision.collection.plan import (
    build_collection_plan,
    collection_status,
    load_collection_protocol,
)
from src.vision.collection.recording import (
    collection_recording_context,
    prepare_collection_scenario,
    validate_collection_recording,
)
from src.vision.collection.dataset import (
    materialize_collection_datasets,
    validate_aggregate_coverage,
)
from src.vision.collection.receipt import (
    write_collection_validation_receipt,
)
from src.vision.collection.pilot import write_pilot_recording
from src.sensors.gazebo_camera import parse_gazebo_image_message


class VisualCollectionRecordingTests(unittest.TestCase):
    def _write_collection_recording(
        self,
        directory,
        plan,
        report_path,
        row=None,
        automatic=True,
    ):
        row = row or plan["scenarios"][0]
        context = collection_recording_context(
            plan,
            row["scenario_id"],
            report_path,
        )
        if automatic:
            Path(directory).mkdir(parents=True, exist_ok=True)
            writer = MissionEventWriter(
                Path(directory, "flight_events.jsonl")
            )
            lifecycle = writer.publish(
                "mission_completed",
                status="completed",
                phase="landed",
                landing_confirmed=True,
            )
            context["flight_lifecycle"] = lifecycle
        simulator_labels = {
            "transformer": 1,
            "switchgear": 2,
            "capacitor_bank": 3,
            "reactor": 4,
        }
        labelled_boxes = [
            box(simulator_labels[class_name], 0, 0, 1, 1)
            for class_name in row["expected_map_class_inventory"]
        ]
        pixels = bytes([255, 0, 0, 0, 255, 0])
        timestamps = tuple(index + 0.1 for index in range(10))
        frames = [
            parse_gazebo_image_message(
                image_message(timestamp, pixels, sequence=index),
                topic="/research_camera/image",
                staging_directory=directory,
                receive_index=index,
            )
            for index, timestamp in enumerate(timestamps, start=1)
        ]
        truths = [
            parse_gazebo_truth_message(
                truth_message(
                    timestamp,
                    [] if index == 10 else labelled_boxes,
                ),
                topic="/research_camera/boxes",
                width=2,
                height=1,
                receive_index=index,
            )
            for index, timestamp in enumerate(timestamps, start=1)
        ]
        events = [
            {"simulation_timestamp": 0.0, "mission_phase": "cruise_distant"},
            {"simulation_timestamp": 3.0, "mission_phase": "approach"},
            {"simulation_timestamp": 5.0, "mission_phase": "close_inspection"},
            {"simulation_timestamp": 8.0, "mission_phase": "target_transition"},
        ]
        write_pilot_recording(
            directory,
            frames,
            truths,
            recording_id=row["recording_id"],
            mission_events=events,
            recording_context_override=context,
        )

    def test_prepared_scenario_records_only_applied_visual_randomization(self):
        plan = build_collection_plan()
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_collection_scenario(
                plan,
                plan["scenarios"][0]["scenario_id"],
                directory,
            )
            report = json.loads(
                Path(prepared["scenario_report"]).read_text()
            )
            self.assertTrue(Path(prepared["world_path"]).is_file())
            self.assertTrue(Path(prepared["planner_path"]).is_file())
            self.assertIn(
                "--visual-mission-events",
                prepared["flight_command"],
            )
            self.assertEqual(
                prepared["flight_events_path"],
                str(
                    Path(prepared["recording_directory"])
                    / "flight_events.jsonl"
                ),
            )
            self.assertIn("visual_scenario_config_hash", report)
            self.assertIn(
                "camera_noise_stddev",
                report["randomization_fields_not_applied_to_visuals"],
            )
            self.assertNotIn(
                "camera_noise_stddev",
                report["applied_visual_randomization_fields"],
            )

    def test_collection_recording_uses_planned_identity_and_validates(self):
        plan = build_collection_plan()
        row = plan["scenarios"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                **row,
                "collection_plan_identity_sha256": plan[
                    "collection_plan_identity_sha256"
                ],
                "config_hash": "1" * 64,
                "visual_scenario_config_hash": "2" * 64,
                "world_sha256": "3" * 64,
                "planner_config_sha256": "4" * 64,
            }
            report_path = root / "scenario.json"
            report_path.write_text(json.dumps(report))
            recording = root / "recording"
            self._write_collection_recording(
                recording,
                plan,
                report_path,
                automatic=True,
            )
            result = validate_collection_recording(recording, plan)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["split"], "train")
            metadata = json.loads(
                Path(recording, "metadata.json").read_text()
            )
            self.assertEqual(
                metadata["recording_type"],
                "labelled_visual_collection",
            )
            self.assertNotIn("pilot_recording_schema_version", metadata)
            self.assertEqual(metadata["dataset_role"], "development")
            identity = json.loads(
                Path(
                    recording,
                    "identity/recording_identity.json",
                ).read_text()
            )
            self.assertIn("flight_events_manifest_sha256", identity)
            with Path(recording, "flight_events.jsonl").open("a") as destination:
                destination.write("{}\n")
            with self.assertRaisesRegex(
                ValueError,
                "flight lifecycle manifest hash",
            ):
                validate_collection_recording(recording, plan)

    def test_aggregate_gate_requires_every_class_in_every_split(self):
        protocol = load_collection_protocol()
        complete = {
            split: {
                "classes": {
                    class_name: 1
                    for class_name in (
                        "transformer",
                        "switchgear",
                        "capacitor_bank",
                        "reactor",
                    )
                },
                "verified_no_target_frames": 1,
            }
            for split in ("train", "validation", "test")
        }
        validate_aggregate_coverage(complete, protocol)
        complete["validation"]["classes"]["reactor"] = 0
        with self.assertRaisesRegex(ValueError, "validation reactor"):
            validate_aggregate_coverage(complete, protocol)

    def test_materialization_creates_isolated_development_and_test_identities(self):
        plan = build_collection_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row in plan["scenarios"]:
                report = {
                    **row,
                    "collection_plan_identity_sha256": plan[
                        "collection_plan_identity_sha256"
                    ],
                    "config_hash": "1" * 64,
                    "visual_scenario_config_hash": object_sha256(row),
                    "world_sha256": "3" * 64,
                    "planner_config_sha256": "4" * 64,
                }
                report_path = root / f"{row['scenario_id']}.json"
                report_path.write_text(json.dumps(report))
                recording = root / "recordings" / row["recording_id"]
                self._write_collection_recording(
                    recording,
                    plan,
                    report_path,
                    row,
                )
            result = materialize_collection_datasets(
                plan,
                root / "recordings",
                root,
            )
            development = result["datasets"]["development"]
            held_out = result["datasets"]["held_out_test"]
            self.assertEqual(development["ordered_frame_count"], 500)
            self.assertEqual(held_out["ordered_frame_count"], 100)
            held_out_identity = json.loads(
                Path(held_out["path"]).read_text()
            )
            self.assertEqual(held_out_identity["dataset_role"], "held_out_test")
            self.assertEqual(held_out_identity["seed_ids"], list(range(2051, 2061)))
            first = plan["scenarios"][0]
            first_root = root / "recordings" / first["recording_id"]
            validation = validate_collection_recording(first_root, plan)
            write_collection_validation_receipt(first_root, plan, validation)
            status = collection_status(plan, root / "recordings")
            self.assertEqual(status["recording_states"]["complete"], 1)
            self.assertEqual(status["recording_states"]["unvalidated"], 59)


if __name__ == "__main__":
    unittest.main()
