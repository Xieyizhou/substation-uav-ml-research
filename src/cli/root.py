"""Top-level command routing for the UAV project."""

from __future__ import annotations

import argparse
import sys

from src.cli import astar, checks, experiments, maintenance, models, reports, sensors
from src.cli.process import run_script


FORWARDED_SCRIPTS = {
    "map": ("scripts/maps/switch_map.py", "Opening map manager"),
    "point": ("scripts/maps/switch_target.py", "Opening destination manager"),
    "target": ("scripts/maps/switch_target.py", "Opening destination manager"),
    "task": ("scripts/flight/run_task.py", "Opening compact task runner"),
}
MODULE_COMMANDS = {
    "astar": astar.main,
    "experiment": experiments.main,
    "report": reports.main,
    "check": checks.main,
    "maintenance": maintenance.main,
    "sensor": sensors.main,
    "model": models.main,
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Unified command center for maps, flight, experiments, and reports.",
        epilog=(
            "Start here: python main.py map; python main.py point; "
            "python main.py task; python main.py check all"
        ),
    )
    commands = parser.add_subparsers(dest="command", metavar="command")
    definitions = (
        ("map", "Select, generate, preview, or start a map"),
        ("point", "Select the destination point", ["target"]),
        ("task", "Run a compact flight task"),
        ("astar", "Use advanced A* preview, flight, and analysis controls"),
        ("experiment", "Run official experiment stages"),
        ("report", "Analyze logs and build experiment reports"),
        ("check", "Run environment and regression checks"),
        ("maintenance", "Run infrequent data maintenance"),
        ("sensor", "Inspect, record, or replay research sensors"),
        ("model", "Train, evaluate, or benchmark research models"),
    )
    for definition in definitions:
        name, help_text, *alias = definition
        commands.add_parser(
            name,
            help=help_text,
            aliases=alias[0] if alias else (),
        )
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 2
    if arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = arguments[0]
    if command not in {*FORWARDED_SCRIPTS, *MODULE_COMMANDS}:
        print(f"Unknown command: {command}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2
    if command in FORWARDED_SCRIPTS:
        script, description = FORWARDED_SCRIPTS[command]
        return run_script(script, arguments[1:], description)
    return MODULE_COMMANDS[command](arguments[1:])
