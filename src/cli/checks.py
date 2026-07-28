"""Project health and regression checks."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from src.cli.process import PROJECT_ROOT, forwarded_args, run_module


def check_environment():
    checks = [
        ("Python 3.9+", sys.version_info >= (3, 9), sys.version.split()[0]),
        ("MAVSDK package", importlib.util.find_spec("mavsdk") is not None, "Python import"),
        ("Gazebo gz command", shutil.which("gz") is not None, shutil.which("gz") or "not found"),
        ("PX4 source", (Path.home() / "PX4-Autopilot").is_dir(), "~/PX4-Autopilot"),
    ]
    print("\nEnvironment Check")
    print("=" * 68)
    for label, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL':<5} {label:<20} {detail}")
    print("=" * 68)
    return 0 if all(passed for _, passed, _ in checks) else 1


def run_tests(map_only=False, arguments=None):
    if map_only:
        suites = [
            "tests.test_map_catalog",
            "tests.test_target_catalog",
            "tests.test_goal_marker",
        ]
        return run_module("unittest", ["-v", *suites], "Checking map and target safety")
    return run_module(
        "unittest",
        ["discover", "-s", "tests", "-v", *forwarded_args(arguments)],
        "Running complete regression test suite",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py check",
        description="Check dependencies, planning helpers, maps, or the complete project.",
    )
    actions = parser.add_subparsers(dest="action")
    actions.add_parser("environment", help="Check local dependencies")
    for name, help_text in (
        ("perception", "Run the perception smoke test"),
        ("replan", "Run the local-replan dry run"),
        ("tests", "Run the complete unit-test suite"),
    ):
        actions.add_parser(name, help=help_text)
    actions.add_parser("maps", help="Check map, target, and marker safety")
    actions.add_parser("all", help="Run environment, perception, and unit checks")
    return parser


def main(argv=None):
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if argv else 2
    action = argv[0]
    valid_actions = {"environment", "perception", "replan", "maps", "tests", "all"}
    if action not in valid_actions:
        print(f"Unknown check action: {action}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2
    if action == "environment":
        return check_environment()
    if action == "perception":
        return run_module(
            "src.flight.test_simple_perception",
            forwarded_args(argv[1:]),
            "Running perception check",
        )
    if action == "replan":
        return run_module(
            "src.logging.test_local_replan",
            forwarded_args(argv[1:]),
            "Running local-replan check",
        )
    if action == "maps":
        return run_tests(map_only=True)
    if action == "tests":
        return run_tests(arguments=argv[1:])

    statuses = [
        check_environment(),
        run_module("src.flight.test_simple_perception", description="Running perception check"),
        run_tests(),
    ]
    return 0 if all(status == 0 for status in statuses) else 1
