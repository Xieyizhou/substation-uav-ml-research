from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.cli import sandbox


HASH = "a" * 64


class SandboxMapCLITests(unittest.TestCase):
    def test_parser_exposes_only_bounded_map_run_modes(self):
        args = sandbox.build_parser().parse_args([
            "map-run", "--map-id", "custom_map", "--revision-id", HASH,
            "--mission-id", "round_trip", "--display-mode", "visual_preview",
            "--record",
        ])
        self.assertEqual(args.display_mode, "visual_preview")
        self.assertTrue(args.record)
        with self.assertRaises(SystemExit):
            sandbox.build_parser().parse_args([
                "map-run", "--map-id", "custom_map", "--revision-id", HASH,
                "--mission-id", "round_trip", "--display-mode", "shell",
            ])

    @patch("src.sandbox.map_cli._command")
    def test_map_run_and_inspection_dispatch_to_fixed_services(self, command):
        runner = Mock(return_value=Path("outputs/sandbox/map_runs/run-1"))
        inspector = Mock(return_value={"latest": {"state": "complete"}})
        command.side_effect = lambda module, name: (
            runner if name == "run_sandbox_map" else inspector
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = sandbox.main([
                "--project-root", str(root), "map-run",
                "--map-id", "custom_map", "--revision-id", HASH,
                "--mission-id", "round_trip",
            ])
            inspected = sandbox.main([
                "--project-root", str(root), "map-run-inspect",
                "--run-id", "run-1",
            ])
        self.assertEqual((result, inspected), (0, 0))
        runner.assert_called_once()
        inspector.assert_called_once()


if __name__ == "__main__":
    unittest.main()
