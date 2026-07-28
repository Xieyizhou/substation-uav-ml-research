"""Versioned research dataset collection, validation, and summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml.dataset_builder import (
    collect_replay,
    summarize_dataset,
    validate_dataset_directory,
)
from src.ml.domain_randomization import (
    load_ranges,
    materialize_planner_config,
    materialize_world,
    sample_manifest,
)
from src.ml.artifacts import write_json
from src.maps.map_catalog import map_by_id, project_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANDOMIZATION = ROOT / "config/perception/domain_randomization.json"


def build_parser():
    parser = argparse.ArgumentParser(description="Research dataset tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Label a synchronized LiDAR replay")
    collect.add_argument("--input", type=Path, required=True)
    collect.add_argument("--telemetry", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--map", dest="map_id", required=True)
    collect.add_argument("--target", default="top_right")
    collect.add_argument("--seed", type=int, required=True)
    collect.add_argument("--randomization", type=Path, default=DEFAULT_RANDOMIZATION)
    validate = commands.add_parser("validate", help="Verify dataset hashes and splits")
    validate.add_argument("--dataset", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="Print dataset statistics")
    summarize.add_argument("--dataset", type=Path, required=True)
    world = commands.add_parser(
        "world", help="Materialize a randomized, launchable Gazebo world"
    )
    world.add_argument("--source", type=Path, required=True)
    world.add_argument("--output", type=Path, required=True)
    world.add_argument("--map", dest="map_id", required=True)
    world.add_argument("--seed", type=int, required=True)
    world.add_argument("--randomization", type=Path, default=DEFAULT_RANDOMIZATION)
    world.add_argument(
        "--planner-output",
        type=Path,
        help="Defaults to OUTPUT with a .planner.json suffix.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            config = load_ranges(args.randomization)
            scenario = sample_manifest(config, map_id=args.map_id, seed=args.seed)
            result = collect_replay(
                args.input,
                args.output,
                map_id=args.map_id,
                target_id=args.target,
                seed=args.seed,
                scenario_manifest=scenario,
                telemetry_path=args.telemetry,
            )
        elif args.command == "validate":
            result = validate_dataset_directory(args.dataset)
        elif args.command == "summarize":
            result = summarize_dataset(args.dataset)
        elif args.command == "world":
            config = load_ranges(args.randomization)
            scenario = sample_manifest(config, map_id=args.map_id, seed=args.seed)
            report_path = args.output.with_suffix(".scenario.json")
            result = materialize_world(
                args.source,
                args.output,
                scenario,
                report_path=report_path,
            )
            entry = map_by_id(args.map_id)
            planner_output = args.planner_output or args.output.with_suffix(
                ".planner.json"
            )
            materialize_planner_config(
                project_path(entry["obstacle_config"]), planner_output, result
            )
            result["oracle_planner_config"] = str(planner_output)
            write_json(report_path, result)
        else:
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Data command failed: {error}")
        return 1
