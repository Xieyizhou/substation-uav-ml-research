import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.cli import visual
from src.ml.artifacts import object_sha256
from src.ml.visual_collection import (
    build_collection_plan,
    collection_status,
    load_collection_plan,
    load_collection_protocol,
    validate_collection_plan,
    write_collection_plan,
)


class VisualCollectionPlanTests(unittest.TestCase):
    def test_frozen_plan_balances_splits_without_seed_leakage(self):
        plan = build_collection_plan()
        self.assertEqual(plan["scenario_count"], 60)
        self.assertEqual(
            plan["split_scenario_counts"],
            {"train": 40, "validation": 10, "test": 10},
        )
        rows = plan["scenarios"]
        self.assertEqual(
            {row["seed"] for row in rows if row["split"] == "train"},
            set(range(2001, 2041)),
        )
        self.assertEqual(
            {row["map_id"] for row in rows if row["split"] == "test"},
            {"extreme"},
        )
        self.assertFalse(
            set(range(1001, 1031)).intersection(row["seed"] for row in rows)
        )
        for split in ("train", "validation", "test"):
            inventory = {
                class_name
                for row in rows
                if row["split"] == split
                for class_name in row["expected_map_class_inventory"]
            }
            self.assertEqual(
                inventory,
                {"transformer", "switchgear", "capacitor_bank", "reactor"},
            )

    def test_plan_identity_rejects_post_hoc_allocation_changes(self):
        plan = build_collection_plan()
        changed = copy.deepcopy(plan)
        changed["scenarios"][0]["target_id"] = "center"
        unsigned = {
            key: value
            for key, value in changed.items()
            if key != "collection_plan_identity_sha256"
        }
        changed["collection_plan_identity_sha256"] = object_sha256(unsigned)
        with self.assertRaisesRegex(ValueError, "frozen allocation"):
            validate_collection_plan(changed)

    def test_plan_round_trip_and_status_are_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory, "collection_plan.json")
            plan = write_collection_plan(plan_path)
            self.assertEqual(load_collection_plan(plan_path), plan)
            status = collection_status(plan, Path(directory, "recordings"))
            self.assertEqual(status["recording_states"]["missing"], 60)
            self.assertEqual(status["recording_states"]["recording"], 0)
            self.assertEqual(status["recording_states"]["failed"], 0)
            self.assertEqual(status["recording_states"]["unvalidated"], 0)
            self.assertEqual(
                status["next_scenario"]["scenario_id"],
                "training-top_right-2001",
            )

    def test_status_distinguishes_active_recording_from_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = build_collection_plan()
            recordings = Path(directory, "recordings")
            first = plan["scenarios"][0]
            first_root = recordings / first["recording_id"]
            first_root.mkdir(parents=True)
            (first_root / "live_status.json").write_text(
                json.dumps(
                    {
                        "recording_id": first["recording_id"],
                        "accepted_frame_count": 100,
                        "last_simulation_timestamp": 12.5,
                    }
                )
            )

            active = collection_status(plan, recordings)
            self.assertEqual(active["recording_states"]["recording"], 1)
            self.assertEqual(active["recording_states"]["partial"], 0)
            self.assertEqual(active["next_scenario"]["state"], "recording")

            (first_root / "live_status.json").unlink()
            partial = collection_status(plan, recordings)
            self.assertEqual(partial["recording_states"]["recording"], 0)
            self.assertEqual(partial["recording_states"]["partial"], 1)
            self.assertEqual(partial["next_scenario"]["state"], "partial")

    def test_status_identifies_failed_flight_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = build_collection_plan()
            recordings = Path(directory, "recordings")
            first = plan["scenarios"][0]
            first_root = recordings / first["recording_id"]
            (first_root / "identity").mkdir(parents=True)
            (first_root / "summary.json").write_text("{}")
            (first_root / "identity/recording_identity.json").write_text("{}")
            (first_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "recording_state": "in_progress",
                        "flight_lifecycle": {
                            "event_type": "mission_failed",
                            "status": "failed",
                        },
                    }
                )
            )

            status = collection_status(plan, recordings)
            self.assertEqual(status["recording_states"]["failed"], 1)
            self.assertEqual(status["recording_states"]["unvalidated"], 0)
            self.assertEqual(status["next_scenario"]["state"], "failed")

    def test_cli_exposes_collection_commands(self):
        help_text = visual.build_parser().format_help()
        for command in (
            "collection-plan",
            "collection-prepare",
            "collection-record",
            "collection-record-validate",
            "collection-materialize",
        ):
            self.assertIn(command, help_text)

    def test_collection_defaults_to_flight_lifecycle_and_confirmed_landing(self):
        recording = load_collection_protocol()["recording"]
        self.assertEqual(recording["default_phase_source"], "flight_lifecycle")
        self.assertEqual(
            recording["automatic_stop_condition"],
            "completed_landed_and_landing_confirmed",
        )
        self.assertEqual(recording["default_post_landing_drain_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
