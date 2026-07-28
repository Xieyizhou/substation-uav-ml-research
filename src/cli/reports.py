"""Flight-log and experiment report commands."""

from __future__ import annotations

import argparse
import sys

from src.cli.process import forwarded_args, run_module
from src.maps.map_catalog import current_map, project_path


MODULES = {
    "analyze": "src.logging.analyze_astar_log",
    "summarize": "src.logging.summarize_experiments",
    "compare": "src.logging.compare_experiment_sets",
    "collect-maps": "src.logging.map_run_collection",
    "validate-active": "src.logging.validate_active_replan_runs",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py report",
        description="Analyze logs and build experiment reports.",
    )
    actions = parser.add_subparsers(dest="action")
    help_text = {
        "analyze": "Analyze one A* log",
        "summarize": "Refresh per-stage summaries",
        "compare": "Compare official experiment stages",
        "collect-maps": "Collect analyzed PX4 run coverage by map",
        "validate-active": "Validate recent active-replan runs",
    }
    for name in MODULES:
        actions.add_parser(name, help=help_text[name])
    return parser


def _analysis_args(arguments):
    arguments = forwarded_args(arguments)
    has_config = any(
        item == "--obstacle-config" or item.startswith("--obstacle-config=")
        for item in arguments
    )
    if "--help" not in arguments and "-h" not in arguments and not has_config:
        arguments.extend(
            ["--obstacle-config", str(project_path(current_map()["obstacle_config"]))]
        )
    return arguments


def main(argv=None):
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if argv else 2
    action = argv[0]
    if action not in MODULES:
        print(f"Unknown report action: {action}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2
    arguments = (
        _analysis_args(argv[1:])
        if action == "analyze"
        else forwarded_args(argv[1:])
    )
    return run_module(
        MODULES[action],
        arguments,
        f"Running report: {action}",
    )
