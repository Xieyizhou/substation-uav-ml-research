"""Advanced A* preview, flight, and log-analysis commands."""

from __future__ import annotations

import argparse
import sys

from src.cli.process import forwarded_args, run_module
from src.maps.map_catalog import current_map, project_path


def _action_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py astar",
        description="Advanced A* controls. Flight options are forwarded to the flight runner.",
    )
    actions = parser.add_subparsers(dest="action")
    for name, help_text in (
        ("preview", "Plan the selected route without starting a flight"),
        ("fly", "Fly the selected route; PX4 must already be running"),
        ("analyze", "Analyze a flight log and refresh its report"),
    ):
        actions.add_parser(name, help=help_text)
    return parser


def _has_option(arguments, option):
    return any(item == option or item.startswith(f"{option}=") for item in arguments)


def _with_selected_map(arguments):
    arguments = forwarded_args(arguments)
    if (
        "--help" not in arguments
        and "-h" not in arguments
        and "--use-built-in-grid" not in arguments
        and not _has_option(arguments, "--obstacle-config")
    ):
        config = project_path(current_map()["obstacle_config"])
        arguments.extend(["--obstacle-config", str(config)])
    return arguments


def main(argv=None):
    parser = _action_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if argv else 2
    action = argv[0]
    if action not in {"preview", "fly", "analyze"}:
        print(f"Unknown A* action: {action}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2
    arguments = _with_selected_map(argv[1:])
    if action == "preview":
        return run_module(
            "src.flight.fly_astar_path",
            ["--dry-run", *arguments],
            "Generating A* route preview",
        )
    if action == "fly":
        return run_module(
            "src.flight.fly_astar_path",
            arguments,
            "Starting A* flight (PX4 must already be running)",
        )
    return run_module(
        "src.logging.analyze_astar_log",
        arguments,
        "Analyzing A* flight log",
    )
