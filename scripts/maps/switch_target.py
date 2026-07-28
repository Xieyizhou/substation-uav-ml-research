#!/usr/bin/env python3
"""Select and inspect per-map A* destination point presets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.maps.map_catalog import MapCatalogError, current_map, map_by_id  # noqa: E402
from src.maps.goal_marker import sync_running_goal_marker  # noqa: E402
from src.maps.target_catalog import (  # noqa: E402
    TargetCatalogError,
    current_target,
    flight_runner_pid,
    select_target,
    target_field_value,
    targets_for_map,
)


def print_target_table(entry, selected_id=None):
    """Print the English destination menu for one map."""
    targets = targets_for_map(entry)
    selected_id = current_target(entry)["id"] if selected_id is None else selected_id
    print(f"\nDestination Points — {entry['id']} / {entry['display_name']}")
    print("=" * 78)
    for index, target in enumerate(targets, start=1):
        marker = "→" if target["id"] == selected_id else " "
        cell = target["cell"]
        print(
            f"{marker} {index}. {target['id']:<10} "
            f"cell=({cell[0]:>2}, {cell[1]:>2})  {target['display_name']}"
        )
        print(f"     {target['description']}")
    print("=" * 78)
    print("→ marks the current destination for this map")


def print_current(entry, target=None):
    """Print the active map and destination point."""
    target = current_target(entry) if target is None else target
    cell = target["cell"]
    print(f"Current map: {entry['id']} — {entry['display_name']}")
    print(f"Current destination: {target['id']} — {target['display_name']}")
    print(f"Grid cell: ({cell[0]}, {cell[1]})")
    print(f"Description: {target['description']}")


def require_flight_stopped():
    """Prevent a destination mutation during an active managed flight."""
    pid = flight_runner_pid()
    if pid is None:
        return True
    print(
        f"Cannot switch destination: a project-managed flight is running (PID {pid}).",
        file=sys.stderr,
    )
    print("Wait for it to land or stop the flight before switching.", file=sys.stderr)
    return False


def report_marker_sync(entry, target):
    """Try to move the live red marker and explain any fallback behavior."""
    status, detail = sync_running_goal_marker(entry, target)
    if status == "updated":
        print("Gazebo red goal marker updated immediately.")
    elif status == "not_running":
        print("Red goal marker will use this point when the map next starts.")
    else:
        print(f"WARNING: live Gazebo marker update was unavailable: {detail}")
        print("The saved point is valid; restart the map to refresh the red marker.")


def choose_interactively(entry):
    """Prompt for a target number or ID."""
    targets = targets_for_map(entry)
    print_target_table(entry)
    try:
        answer = input(
            "Select a destination number or ID (press Enter to cancel): "
        ).strip()
    except EOFError:
        print("No interactive input is available. Use: python main.py point use <id>")
        return 2
    if not answer:
        print("Cancelled.")
        return 0
    if answer.isdigit() and 1 <= int(answer) <= len(targets):
        target_id = targets[int(answer) - 1]["id"]
    else:
        target_id = answer
    if not require_flight_stopped():
        return 1
    _, target = select_target(target_id, entry["id"])
    print("\nDestination selection updated. PX4 does not need to be restarted.")
    print_current(entry, target)
    report_marker_sync(entry, target)
    print("\nNext step: python main.py astar preview --return-home")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Select an A* destination point. Run without arguments for the "
            "interactive English menu."
        )
    )
    parser.add_argument(
        "--map",
        dest="map_id",
        help="Operate on this map instead of the current map.",
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List destination points")
    list_parser.add_argument("--json", action="store_true", help="Output JSON")

    current_parser = subparsers.add_parser("current", help="Show the current point")
    current_parser.add_argument("--field", help="Print one target field")

    use_parser = subparsers.add_parser("use", help="Select a destination point")
    use_parser.add_argument("target_id")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        entry = map_by_id(args.map_id) if args.map_id else current_map()
        if args.command is None:
            return choose_interactively(entry)
        if args.command == "list":
            targets = targets_for_map(entry)
            if args.json:
                print(json.dumps(targets, ensure_ascii=False, indent=2))
            else:
                print_target_table(entry)
            return 0
        if args.command == "current":
            target = current_target(entry)
            if args.field:
                print(target_field_value(target, args.field))
            else:
                print_current(entry, target)
            return 0
        if args.command == "use":
            if not require_flight_stopped():
                return 1
            _, target = select_target(args.target_id, entry["id"])
            print("Destination selection updated. PX4 does not need to be restarted.")
            print_current(entry, target)
            report_marker_sync(entry, target)
            return 0
    except (MapCatalogError, TargetCatalogError) as error:
        print(f"Destination configuration error: {error}", file=sys.stderr)
        return 2
    parser.error("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
