"""Local research sandbox inspection commands."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

from src.inspection.config import InspectionConfig
from src.sandbox.profile_cli import handle_profile_command, register_profile_commands
from src.sandbox.profiles import PROFILE_NAMES
from src.sandbox.map_cli import handle_map_command, register_map_commands
from src.vision.collection.display import DISPLAY_MODES
from src.sandbox.storage_cli import handle_storage_command, register_storage_commands
from src.sandbox.workbench_cli import (
    handle_workbench_command, register_workbench_commands,
)


def _command(module, name):
    return getattr(import_module(module), name)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py sandbox",
        description="Inspect and safely operate the local UAV research sandbox.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="development")
    commands = parser.add_subparsers(dest="command", required=True)
    register_profile_commands(commands)
    register_storage_commands(commands)
    register_workbench_commands(commands)
    register_map_commands(commands)
    serve = commands.add_parser("serve", help="Run the controlled local sandbox app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    smoke = commands.add_parser(
        "flight-smoke",
        help="Run one real visual flight without saving dataset frames",
    )
    smoke.add_argument(
        "--plan",
        type=Path,
        default=Path("data/research/visual_collection_v2/collection_plan.json"),
    )
    smoke.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/research/visual_collection_v2"),
    )
    smoke.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/sandbox/flight_smoke"),
    )
    smoke.add_argument("--scenario-id")
    smoke.add_argument("--startup-timeout", type=float, default=180.0)
    smoke.add_argument("--probe-timeout", type=float, default=5.0)
    smoke.add_argument("--flight-timeout", type=float)
    smoke.add_argument(
        "--display-mode", choices=DISPLAY_MODES, default="headless",
        help="Run Gazebo without a window or show its visual preview",
    )
    recipe = commands.add_parser(
        "recipe-create", help="Create an identity-bound non-blind visual recipe"
    )
    recipe.add_argument("--name", required=True)
    recipe.add_argument(
        "--partition", choices=["validation", "full_validation"],
        default="validation",
    )
    recipe.add_argument("--input-size", type=int, choices=[320, 416, 640], default=416)
    recipe.add_argument("--frame-skip", type=int, choices=[1, 2, 3], default=1)
    recipe.add_argument("--frame-limit", type=int)
    inspect = commands.add_parser(
        "recipe-inspect", help="Validate a sandbox visual experiment recipe"
    )
    inspect.add_argument("--input", type=Path, required=True)
    run = commands.add_parser(
        "experiment-run", help="Run one identity-bound non-blind visual recipe"
    )
    run.add_argument("--recipe", type=Path, required=True)
    result = commands.add_parser(
        "experiment-inspect", help="Validate a completed sandbox experiment result"
    )
    result.add_argument("--input", type=Path, required=True)
    supervisor = commands.add_parser(
        "supervisor-gate", help="Verify safe stop and failure diagnostics offline"
    )
    supervisor.add_argument("--output", type=Path, required=True)
    supervisor_inspect = commands.add_parser(
        "supervisor-gate-inspect", help="Validate a supervisor gate receipt"
    )
    supervisor_inspect.add_argument("--input", type=Path, required=True)
    acceptance = commands.add_parser(
        "acceptance-run", help="Run the evidence-bound Sandbox v1 gate"
    )
    acceptance.add_argument("--output", type=Path, required=True)
    acceptance_inspect = commands.add_parser(
        "acceptance-inspect", help="Validate a Sandbox v1 acceptance result"
    )
    acceptance_inspect.add_argument("--input", type=Path, required=True)
    return parser


def _print(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = InspectionConfig.for_profile(args.project_root, args.profile)
    try:
        storage = handle_storage_command(args, config)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"Sandbox storage failed: {error}")
        return 1
    if storage is not None:
        _print(storage)
        return 0
    try:
        profile = handle_profile_command(args, config)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Sandbox profile command failed: {error}")
        return 1
    if profile is not None:
        value, return_code = profile
        _print(value)
        return return_code
    try:
        workbench = handle_workbench_command(args, config)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Sandbox workbench failed: {error}")
        return 1
    if workbench is not None:
        _print(workbench)
        return 0
    map_command = handle_map_command(args, config)
    if map_command is not None:
        value, return_code = map_command
        _print(value)
        return return_code
    if args.command == "doctor":
        InspectionService = _command("src.inspection.service", "InspectionService")
        service = InspectionService(config)
        checks = service.doctor()
        _print(checks)
        return 1 if any(item["status"] == "failure" for item in checks) else 0
    if args.command == "status":
        InspectionService = _command("src.inspection.service", "InspectionService")
        service = InspectionService(config)
        _print({"collection": service.dashboard(), "runtime": service.runtime()})
        return 0
    if args.command == "challenge-run":
        run_challenge_gate = _command(
            "src.sandbox.challenge_gate", "run_challenge_gate"
        )
        try:
            result = run_challenge_gate(config.project_root, args.model_id)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            print(f"Sandbox challenge failed: {error}")
            return 1
        _print(result)
        return 0 if result["challenge_receipt"]["passed"] else 1
    if args.command == "flight-smoke":
        run_flight_smoke = _command("src.sandbox.flight_smoke", "run_flight_smoke")
        try:
            root = run_flight_smoke(
                args.plan,
                args.output_root,
                args.run_root,
                scenario_id=args.scenario_id,
                startup_timeout_s=args.startup_timeout,
                probe_timeout_s=args.probe_timeout,
                flight_timeout_s=args.flight_timeout,
                display_mode=args.display_mode,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Sandbox flight smoke failed: {error}")
            return 1
        _print({"status": "complete", "run_root": str(root)})
        return 0
    if args.command == "recipe-create":
        materialize_recipe = _command(
            "src.sandbox.experiment_recipe", "materialize_recipe"
        )
        try:
            recipe, output = materialize_recipe(
                config.project_root,
                args.name,
                partition=args.partition,
                input_size=args.input_size,
                frame_skip_interval=args.frame_skip,
                frame_limit=args.frame_limit,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(f"Sandbox recipe failed: {error}")
            return 1
        _print({"path": str(output / "recipe.json"), "recipe": recipe.to_record()})
        return 0
    if args.command == "recipe-inspect":
        inspect_recipe = _command("src.sandbox.experiment_recipe", "inspect_recipe")
        try:
            result = inspect_recipe(args.input)
        except (FileNotFoundError, TypeError, ValueError) as error:
            print(f"Sandbox recipe failed: {error}")
            return 1
        _print(result)
        return 0
    if args.command == "experiment-run":
        run_recipe = _command("src.sandbox.experiment_runner", "run_recipe")
        try:
            result = run_recipe(config.project_root, args.recipe)
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
            print(f"Sandbox experiment failed: {error}")
            return 1
        _print(result)
        return 0
    if args.command == "experiment-inspect":
        inspect_result = _command("src.sandbox.experiment_runner", "inspect_result")
        try:
            result = inspect_result(args.input)
        except (FileNotFoundError, TypeError, ValueError) as error:
            print(f"Sandbox experiment failed: {error}")
            return 1
        _print(result)
        return 0
    if args.command == "supervisor-gate":
        run_supervisor_gate = _command(
            "src.sandbox.supervisor_gate", "run_supervisor_gate"
        )
        try:
            result = run_supervisor_gate(config.project_root, args.output)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            print(f"Sandbox supervisor gate failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    if args.command == "supervisor-gate-inspect":
        inspect_supervisor_gate = _command(
            "src.sandbox.supervisor_gate", "inspect_supervisor_gate"
        )
        try:
            result = inspect_supervisor_gate(args.input)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            print(f"Sandbox supervisor gate failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    if args.command == "acceptance-run":
        run_acceptance = _command("src.sandbox.acceptance", "run_acceptance")
        try:
            result = run_acceptance(config.project_root, args.output)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            print(f"Sandbox acceptance failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    if args.command == "acceptance-inspect":
        inspect_acceptance = _command("src.sandbox.acceptance", "inspect_acceptance")
        try:
            result = inspect_acceptance(args.input)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            print(f"Sandbox acceptance failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    serve_inspector = _command("src.inspection.app", "main")
    return serve_inspector([
        "--host", args.host,
        "--port", str(args.port),
        "--project-root", str(args.project_root),
        "--profile", args.profile,
    ])
