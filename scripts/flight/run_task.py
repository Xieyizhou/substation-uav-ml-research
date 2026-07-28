#!/usr/bin/env python3
"""English menu and command-line runner for compact UAV flight tasks."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flight.task_presets import TASKS, task_by_id  # noqa: E402


def run_task(task_id, extra_args=None):
    """Lazy-load the flight stack only after the user selects a task."""
    from src.flight.task_runner import run_task as execute_task

    return execute_task(task_id, extra_args)


def print_tasks():
    print("\nAvailable Flight Tasks")
    print("=" * 72)
    for index, (task_id, preset) in enumerate(TASKS.items(), start=1):
        print(f"  {index}. {task_id:<20} {preset['display_name']}")
        print(f"     {preset['description']}")
    print("=" * 72)


def choose_interactively():
    print_tasks()
    try:
        answer = input("Select a task number or ID (press Enter to cancel): ").strip()
    except EOFError:
        print("No interactive input is available. Use: python main.py task run <id>")
        return 2
    if not answer:
        print("Cancelled.")
        return 0
    task_ids = list(TASKS)
    task_id = task_ids[int(answer) - 1] if answer.isdigit() else answer
    task_by_id(task_id)
    return run_task(task_id)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a compact flight task for the selected map and point."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="List available tasks")
    run_parser = subparsers.add_parser("run", help="Run one task")
    run_parser.add_argument("task_id")
    run_parser.add_argument(
        "task_args",
        nargs=argparse.REMAINDER,
        help="Advanced flight overrides after an optional -- separator.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command is None:
            return choose_interactively()
        if args.command == "list":
            print_tasks()
            return 0
        extra_args = args.task_args
        if extra_args[:1] == ["--"]:
            extra_args = extra_args[1:]
        return run_task(args.task_id, extra_args)
    except (IndexError, ValueError) as error:
        print(f"Task configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
