"""Controlled custom-map and planning command boundary."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from src.vision.collection.display import DISPLAY_MODES


def _command(module, name):
    return getattr(import_module(module), name)


def register_map_commands(commands):
    map_run = commands.add_parser(
        "map-run", help="Run one validated custom-map revision",
    )
    map_run.add_argument("--map-id", required=True)
    map_run.add_argument("--revision-id", required=True)
    map_run.add_argument("--mission-id", required=True)
    map_run.add_argument(
        "--display-mode", choices=DISPLAY_MODES, default="headless",
    )
    map_run.add_argument("--startup-timeout", type=float, default=180.0)
    map_run.add_argument("--flight-timeout", type=float)
    map_run.add_argument("--record", action="store_true")
    inspect = commands.add_parser(
        "map-run-inspect", help="Inspect a completed custom-map run",
    )
    inspect.add_argument("--run-id", required=True)
    semantic = commands.add_parser(
        "semantic-inspection-plan",
        help="Convert one stable visual-depth observation into a gated route",
    )
    semantic.add_argument("--map", type=Path, required=True)
    semantic.add_argument("--observation", type=Path, required=True)
    semantic.add_argument("--output", type=Path, required=True)
    height = commands.add_parser(
        "height-layer-plan",
        help="Plan a deterministic route through explicit altitude layers",
    )
    height.add_argument("--map", type=Path, required=True)
    height.add_argument("--mission-id", required=True)
    height.add_argument("--output", type=Path, required=True)
    height.add_argument(
        "--layers", type=float, nargs="+", default=[1.5, 3.0, 5.0],
    )


def _run_map(args, config):
    run = _command("src.sandbox.map_run", "run_sandbox_map")
    root = run(
        config.project_root, config.sandbox_maps_root,
        config.sandbox_map_runs_root, args.map_id, args.revision_id,
        args.mission_id, display_mode=args.display_mode,
        startup_timeout_s=args.startup_timeout,
        flight_timeout_s=args.flight_timeout, record=args.record,
    )
    return {"status": "complete", "run_root": str(root)}


def handle_map_command(args, config):
    try:
        if args.command == "map-run":
            return _run_map(args, config), 0
        if args.command == "map-run-inspect":
            inspect = _command("src.inspection.map_runs", "inspect_map_run")
            return inspect(config, args.run_id), 0
        if args.command == "semantic-inspection-plan":
            materialize = _command(
                "src.planner.planning_artifacts",
                "materialize_semantic_inspection",
            )
            return materialize(args.map, args.observation, args.output), 0
        if args.command == "height-layer-plan":
            materialize = _command(
                "src.planner.planning_artifacts", "materialize_height_layer_plan",
            )
            return materialize(
                args.map, args.mission_id, args.output,
                layer_altitudes_m=args.layers,
            ), 0
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        return {"error": str(error)}, 1
    return None
