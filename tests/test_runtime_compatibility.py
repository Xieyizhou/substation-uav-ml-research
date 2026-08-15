import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from src.inspection.config import InspectionConfig
from src.runtime_compatibility import inspect_runtime, load_manifest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_tracked_manifest_has_explicit_compatibility_families(self):
        value = load_manifest(ROOT)
        self.assertEqual(value["python"]["minimum_version"], "3.11")
        self.assertEqual(value["gazebo"]["expected_distribution"], "Harmonic")
        self.assertEqual(value["opencv"]["expected_major_version"], 4)
        self.assertEqual(value["qt"]["expected_major_version"], 5)

    def test_gazebo_skips_unsupported_path_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = self._executable(root / "wrong/gz")
            right = self._executable(root / "right/gz")
            px4 = self._px4(root)
            environment = self._environment(root, px4, f"{wrong.parent}:{right.parent}")

            def command(command):
                if command[-2:] == ["sim", "--versions"]:
                    return "7.9.0" if command[0] == str(wrong) else "8.14.0"
                if "rev-parse" in command:
                    return "6e569d87b977d0579966a4cae78abc275aa9aaf3"
                if "describe" in command:
                    return "v1.17.0"
                return None

            with patch("src.runtime_compatibility._run", side_effect=command):
                value = inspect_runtime(ROOT, "development", environment)
        gazebo = next(item for item in value["components"]
                       if item["component_id"] == "gazebo")
        self.assertEqual(gazebo["path"], str(right.resolve()))
        self.assertEqual(gazebo["status"], "compatible")

    def test_formal_blocks_untested_px4_but_development_warns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gz = self._executable(root / "bin/gz")
            px4 = self._px4(root)
            environment = self._environment(root, px4, str(gz.parent))

            def command(command):
                if command[-2:] == ["sim", "--versions"]:
                    return "8.14.0"
                if "rev-parse" in command:
                    return "0000000000000000000000000000000000000000"
                if "describe" in command:
                    return "custom"
                return None

            with patch("src.runtime_compatibility._run", side_effect=command):
                development = inspect_runtime(ROOT, "development", environment)
                formal = inspect_runtime(ROOT, "formal", environment)
        self.assertTrue(development["ready"])
        self.assertEqual(development["overall"], "compatible_with_warning")
        self.assertFalse(formal["ready"])

    def test_inspection_config_honors_verified_px4_root(self):
        selected = "/tmp/selected-px4-runtime"
        with patch.dict(os.environ, {"PX4_ROOT": selected}):
            config = InspectionConfig.for_profile(ROOT, "development")
        self.assertEqual(config.px4_root, Path(selected))

    def test_missing_explicit_formula_paths_do_not_validate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gz = self._executable(root / "bin/gz")
            px4 = self._px4(root)
            environment = self._environment(root, px4, str(gz.parent))
            environment["UAV_SANDBOX_OPENCV_PREFIX"] = str(root / "missing-opencv")
            environment["UAV_SANDBOX_QT_PREFIX"] = str(root / "missing-qt")

            def command(command):
                if command[-2:] == ["sim", "--versions"]:
                    return "8.14.0"
                if "rev-parse" in command:
                    return "6e569d87b977d0579966a4cae78abc275aa9aaf3"
                if "describe" in command:
                    return "v1.17.0"
                return None

            with patch("src.runtime_compatibility._run", side_effect=command):
                value = inspect_runtime(ROOT, "development", environment)
        statuses = {item["component_id"]: item["status"] for item in value["components"]}
        self.assertEqual(statuses["opencv"], "missing")
        self.assertEqual(statuses["qt"], "missing")

    @staticmethod
    def _executable(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    @staticmethod
    def _px4(root):
        px4 = root / "PX4-Autopilot"
        for relative in (".git/HEAD", "Makefile", "CMakeLists.txt", "Tools/simulation/gz/.keep"):
            path = px4 / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return px4

    @staticmethod
    def _environment(root, px4, path):
        opencv = root / "opencv@4"
        qt = root / "qt@5"
        opencv.mkdir()
        qt.mkdir()
        return {
            "PATH": path,
            "PX4_ROOT": str(px4),
            "UAV_SANDBOX_OPENCV_PREFIX": str(opencv),
            "UAV_SANDBOX_QT_PREFIX": str(qt),
        }


if __name__ == "__main__":
    unittest.main()
