#!/usr/bin/env python3
"""Select, inspect, preview, or start a catalogued substation test map."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.maps.map_catalog import (  # noqa: E402
    MapCatalogError,
    current_map,
    field_value,
    list_maps,
    map_by_id,
    project_path,
    px4_launcher_pid,
    select_map,
    spawn_pose_text,
)
from src.maps.target_catalog import flight_runner_pid  # noqa: E402


def difficulty_bar(level):
    return "●" * level + "○" * (5 - level)


def print_map_table(entries=None, selected_id=None):
    entries = list_maps() if entries is None else entries
    selected_id = current_map()["id"] if selected_id is None else selected_id
    print("\nAvailable Substation Maps")
    print("=" * 78)
    for index, entry in enumerate(entries, start=1):
        marker = "→" if entry["id"] == selected_id else " "
        print(
            f"{marker} {index}. {entry['id']:<9} "
            f"{difficulty_bar(entry['difficulty'])} "
            f"{entry['difficulty_label']:<4}  {entry['display_name']}"
        )
        print(f"     {entry['description']}")
    print("=" * 78)
    print("→ marks the current selection")


def print_current(entry=None):
    entry = current_map() if entry is None else entry
    print(f"Current map: {entry['id']} — {entry['display_name']}")
    print(f"Difficulty: {entry['difficulty_label']} ({entry['difficulty']}/5)")
    print(f"Description: {entry['description']}")
    print(f"Gazebo world: {entry['world_name']}")
    print(f"World file: {entry['world_file']}")
    print(f"A* config: {entry['obstacle_config']}")
    print(f"PX4 spawn pose: {spawn_pose_text(entry)}")
    pid = px4_launcher_pid()
    print(f"PX4 status: {'running, PID ' + str(pid) if pid else 'not running'}")


def require_project_idle(action):
    flight_pid = flight_runner_pid()
    if flight_pid is not None:
        print(
            f"Cannot {action}: a project-managed flight is running (PID {flight_pid}).",
            file=sys.stderr,
        )
        print("Wait for it to land or stop the flight first.", file=sys.stderr)
        return False
    px4_pid = px4_launcher_pid()
    if px4_pid is not None:
        print(
            f"Cannot {action}: the project-managed PX4 launcher is running (PID {px4_pid}).",
            file=sys.stderr,
        )
        print("Press Ctrl+C in the PX4 terminal and wait for it to exit.", file=sys.stderr)
        return False
    return True


def choose_interactively():
    entries = list_maps()
    print_map_table(entries)
    try:
        answer = input("Select a map number or ID (press Enter to cancel): ").strip()
    except EOFError:
        print("No interactive input is available. Use: python main.py map use <map-id>")
        return 2
    if not answer:
        print("Cancelled.")
        return 0
    if answer.isdigit() and 1 <= int(answer) <= len(entries):
        map_id = entries[int(answer) - 1]["id"]
    else:
        map_id = answer
    if not require_project_idle("switch maps"):
        return 1
    entry = select_map(map_id)
    print("\nMap selection updated.")
    print_current(entry)
    print("\nNext step: python main.py map start")
    return 0


def run_preview(entry, return_home=False):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "astar",
        "preview",
        "--obstacle-config",
        str(project_path(entry["obstacle_config"])),
    ]
    if return_home:
        command.append("--return-home")
    print(f"Generating a preview for {entry['id']} — {entry['display_name']}...")
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def launcher_environment(entry, vehicle_model="x500", display_mode="headless"):
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "MAP_ID": entry["id"],
            "WORLD_NAME": entry["world_name"],
            "WORLD_SRC": str(project_path(entry["world_file"])),
            "PX4_GZ_MODEL_POSE": spawn_pose_text(entry),
            "OBSTACLE_CONFIG": entry["obstacle_config"],
            "SIM_MODEL": vehicle_model,
            "UAV_SANDBOX_DISPLAY_MODE": display_mode,
            "GZ_IP": "127.0.0.1",
            "GZ_PARTITION": "substation_uav",
        }
    )
    return environment


def check_map(entry, vehicle_model="x500"):
    print(
        f"Checking map launcher for {entry['id']} — {entry['display_name']}...",
        flush=True,
    )
    return subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "flight" / "start_px4_substation.sh"),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        env=launcher_environment(entry, vehicle_model),
    ).returncode


def start_map(entry, vehicle_model="x500", display_mode="headless"):
    if not require_project_idle("start a new map"):
        return 1
    entry = select_map(entry["id"])
    print(f"Starting map {entry['id']} — {entry['display_name']}...")
    try:
        return subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts" / "flight" / "start_px4_substation.sh")],
            cwd=PROJECT_ROOT,
            env=launcher_environment(entry, vehicle_model, display_mode),
        ).returncode
    except KeyboardInterrupt:
        print("\nPX4/Gazebo stopped.", file=sys.stderr)
        return 130


def generate_maps():
    if not require_project_idle("regenerate maps"):
        return 1
    generator = PROJECT_ROOT / "scripts" / "maps" / "generate_test_maps.py"
    print("Regenerating catalogued Gazebo worlds and A* map configs...")
    status = subprocess.run([sys.executable, str(generator)], cwd=PROJECT_ROOT).returncode
    if status == 0:
        print(f"Generated and validated {len(list_maps())} catalogued maps.")
    return status


def build_parser():
    parser = argparse.ArgumentParser(
        description="Select, preview, or start a substation test map. Run without arguments for interactive selection."
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List all maps")
    list_parser.add_argument("--json", action="store_true", help="Output JSON")

    current_parser = subparsers.add_parser("current", help="Show the current map")
    current_parser.add_argument("--field", help="Print one field for script integration")

    use_parser = subparsers.add_parser("use", help="Select a map without starting PX4")
    use_parser.add_argument("map_id")

    preview_parser = subparsers.add_parser("preview", help="Generate an A* preview for a map")
    preview_parser.add_argument("map_id", nargs="?", help="Defaults to the current map")
    preview_parser.add_argument("--return-home", action="store_true")

    start_parser = subparsers.add_parser("start", help="Select and start a Gazebo/PX4 map")
    start_parser.add_argument("map_id", nargs="?", help="Defaults to the current map")
    start_parser.add_argument(
        "--vehicle-model",
        choices=["x500", "x500_lidar_2d", "x500_research"],
        default="x500",
        help="PX4 Gazebo vehicle model. Default: x500",
    )
    start_parser.add_argument(
        "--display-mode",
        choices=["headless", "visual_preview"],
        default="headless",
    )
    check_parser = subparsers.add_parser(
        "check", help="Validate the selected map launcher without starting PX4"
    )
    check_parser.add_argument("map_id", nargs="?", help="Defaults to the current map")
    check_parser.add_argument(
        "--vehicle-model",
        choices=["x500", "x500_lidar_2d", "x500_research"],
        default="x500",
    )
    subparsers.add_parser("generate", help="Regenerate all catalogued map files")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command is None:
            return choose_interactively()
        if args.command == "list":
            entries = list_maps()
            if args.json:
                print(json.dumps(entries, ensure_ascii=False, indent=2))
            else:
                print_map_table(entries)
            return 0
        if args.command == "current":
            entry = current_map()
            if args.field:
                print(field_value(entry, args.field))
            else:
                print_current(entry)
            return 0
        if args.command == "use":
            if not require_project_idle("switch maps"):
                return 1
            entry = select_map(args.map_id)
            print("Map selection updated.")
            print_current(entry)
            return 0
        if args.command == "preview":
            entry = map_by_id(args.map_id) if args.map_id else current_map()
            return run_preview(entry, args.return_home)
        if args.command == "start":
            entry = map_by_id(args.map_id) if args.map_id else current_map()
            return start_map(entry, args.vehicle_model, args.display_mode)
        if args.command == "check":
            entry = map_by_id(args.map_id) if args.map_id else current_map()
            return check_map(entry, args.vehicle_model)
        if args.command == "generate":
            return generate_maps()
    except MapCatalogError as error:
        print(f"Map configuration error: {error}", file=sys.stderr)
        return 2
    parser.error("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
