import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.logging import map_run_collection


class MapRunCollectionTests(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {
                "id": "complex",
                "display_name": "Complex",
                "difficulty": 4,
                "obstacle_config": "complex.json",
                "world_name": "substation_complex",
            },
            {
                "id": "extreme",
                "display_name": "Extreme",
                "difficulty": 5,
                "obstacle_config": "extreme.json",
                "world_name": "substation_extreme",
            },
        ]

    def test_coverage_normalizes_config_map_names_and_marks_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "complex.json").write_text(
                json.dumps({"map_name": "substation_complex_v1"})
            )
            (root / "extreme.json").write_text(
                json.dumps({"map_name": "substation_extreme_v1"})
            )
            runs = [
                {
                    "map_name": "substation_complex_v2",
                    "run_id": "as_2",
                    "stage": "active_replan",
                    "completed_or_failed": "completed",
                    "final_status": "PASS",
                },
                {
                    "map_name": "substation_complex_v1",
                    "run_id": "as_1",
                    "stage": "static_astar",
                    "completed_or_failed": "completed",
                    "final_status": "PASS",
                },
            ]
            with patch.object(
                map_run_collection, "project_path", side_effect=lambda value: root / value
            ):
                rows = map_run_collection.coverage_rows(
                    self.entries, runs, minimum_runs=2
                )

        self.assertEqual(rows[0]["analyzed_run_count"], 2)
        self.assertEqual(rows[0]["latest_run_id"], "as_2")
        self.assertEqual(rows[0]["stages"], "active_replan,static_astar")
        self.assertEqual(rows[0]["coverage_status"], "COMPLETE")
        self.assertEqual(rows[1]["coverage_status"], "MISSING")

    def test_collection_writes_machine_and_human_readable_outputs(self):
        rows = [
            {
                "map_id": "complex",
                "map_name": "Complex",
                "difficulty": 4,
                "analyzed_run_count": 1,
                "completed_count": 1,
                "pass_count": 1,
                "stages": "static_astar",
                "latest_run_id": "as_1",
                "coverage_status": "COMPLETE",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                patch.object(map_run_collection, "list_maps", return_value=self.entries),
                patch.object(map_run_collection, "find_map_runs", return_value=[]),
                patch.object(map_run_collection, "coverage_rows", return_value=rows),
            ):
                result, csv_path, md_path = map_run_collection.collect_map_coverage(
                    output_dir=output_dir
                )
            self.assertEqual(result, rows)
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            with csv_path.open() as source:
                self.assertEqual(list(csv.DictReader(source))[0]["map_id"], "complex")
            self.assertIn("Only analyzed PX4/Gazebo telemetry", md_path.read_text())


if __name__ == "__main__":
    unittest.main()
