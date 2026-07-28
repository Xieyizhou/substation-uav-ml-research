import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.maps import switch_map
from scripts.maps.prepare_research_vehicle import prepare_world_with_vehicle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "flight" / "start_px4_substation.sh"


class Px4LauncherPathTests(unittest.TestCase):
    def test_launcher_prefers_opencv4_and_refreshes_px4_configuration(self):
        launcher_text = LAUNCHER.read_text()
        self.assertIn("brew --prefix opencv@4", launcher_text)
        self.assertIn("brew install opencv@4", launcher_text)
        self.assertIn('-DOpenCV_DIR="$OpenCV_DIR"', launcher_text)
        self.assertIn("-U 'GSTREAMER_*'", launcher_text)

    def test_research_vehicle_uses_registered_x500_airframe_target(self):
        launcher_text = LAUNCHER.read_text()
        self.assertIn('export PX4_GZ_MODEL_NAME="x500_research_0"', launcher_text)
        self.assertIn('MAKE_SIM_MODEL="x500"', launcher_text)
        self.assertIn('"gz_$MAKE_SIM_MODEL"', launcher_text)

    def test_runtime_world_includes_research_vehicle_at_spawn_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.sdf"
            output = Path(temporary) / "runtime.sdf"
            source.write_text(
                '<sdf version="1.9"><world name="test"></world></sdf>',
                encoding="utf-8",
            )
            prepare_world_with_vehicle(
                source,
                output,
                model_name="x500_research",
                entity_name="x500_research_0",
                pose="-10,-8,0,0,0,1.57",
            )
            runtime = output.read_text(encoding="utf-8")
        self.assertIn("model://x500_research", runtime)
        self.assertIn("<name>x500_research_0</name>", runtime)
        self.assertIn("<pose>-10 -8 0 0 0 1.57</pose>", runtime)

    def make_fake_environment(self, root):
        px4_root = root / "PX4-Autopilot"
        px4_root.mkdir()
        (px4_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")

        bin_dir = root / "bin"
        bin_dir.mkdir()
        for command in ("gz", "make"):
            executable = bin_dir / command
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

        environment = os.environ.copy()
        environment.pop("PROJECT_ROOT", None)
        environment["PX4_ROOT"] = str(px4_root)
        environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
        return environment

    def test_check_derives_project_root_from_script_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = self.make_fake_environment(root)
            unrelated_cwd = root / "unrelated"
            unrelated_cwd.mkdir()

            result = subprocess.run(
                ["bash", str(LAUNCHER), "--check"],
                cwd=unrelated_cwd,
                env=environment,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Project root: {PROJECT_ROOT}", result.stdout)
        self.assertIn("Preflight check passed", result.stdout)
        self.assertNotIn("Map ID:       custom", result.stdout)

    def test_stale_project_root_override_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = self.make_fake_environment(root)
            environment["PROJECT_ROOT"] = str(root / "missing-project")

            result = subprocess.run(
                ["bash", str(LAUNCHER), "--check"],
                env=environment,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Project root: {PROJECT_ROOT}", result.stdout)
        self.assertIn("ignoring stale PROJECT_ROOT", result.stderr)

    def test_map_manager_overrides_stale_project_root(self):
        entry = {
            "id": "simple",
            "display_name": "Simple Distribution Station",
            "world_name": "substation_simple",
            "world_file": "simulation/worlds/substation_simple.sdf",
            "obstacle_config": "config/maps/substation_simple.json",
            "spawn_pose": [-10, -10, 0, 0, 0, 0],
        }
        with mock.patch.object(switch_map, "require_project_idle", return_value=True), mock.patch.object(
            switch_map, "select_map", return_value=entry
        ), mock.patch.object(switch_map.subprocess, "run") as run:
            run.return_value.returncode = 0
            with mock.patch.dict(os.environ, {"PROJECT_ROOT": "/old/drone-ai"}):
                result = switch_map.start_map(entry)

        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.kwargs["env"]["PROJECT_ROOT"], str(PROJECT_ROOT))

    def test_map_manager_treats_control_c_as_clean_user_stop(self):
        entry = {
            "id": "simple",
            "display_name": "Simple Distribution Station",
            "world_name": "substation_simple",
            "world_file": "simulation/worlds/substation_simple.sdf",
            "obstacle_config": "config/maps/substation_simple.json",
            "spawn_pose": [-10, -10, 0, 0, 0, 0],
        }
        with mock.patch.object(
            switch_map, "require_project_idle", return_value=True
        ), mock.patch.object(
            switch_map, "select_map", return_value=entry
        ), mock.patch.object(
            switch_map.subprocess, "run", side_effect=KeyboardInterrupt
        ):
            result = switch_map.start_map(entry)
        self.assertEqual(result, 130)


if __name__ == "__main__":
    unittest.main()
