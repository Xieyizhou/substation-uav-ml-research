"""Local research sandbox inspection commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inspection import InspectionConfig, InspectionService
from src.inspection.app import main as serve_inspector
from src.sandbox.acceptance import inspect_acceptance, run_acceptance
from src.sandbox.experiment_recipe import inspect_recipe, materialize_recipe
from src.sandbox.experiment_runner import inspect_result, run_recipe
from src.sandbox.flight_smoke import run_flight_smoke
from src.sandbox.supervisor_gate import inspect_supervisor_gate, run_supervisor_gate


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py sandbox",
        description="Inspect and safely operate the local UAV research sandbox.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check dependencies, paths, and disk space")
    commands.add_parser("status", help="Show collection and runtime status")
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
    config = InspectionConfig.defaults(args.project_root)
    service = InspectionService(config)
    if args.command == "doctor":
        checks = service.doctor()
        _print(checks)
        return 1 if any(item["status"] == "failure" for item in checks) else 0
    if args.command == "status":
        _print({"collection": service.dashboard(), "runtime": service.runtime()})
        return 0
    if args.command == "flight-smoke":
        try:
            root = run_flight_smoke(
                args.plan,
                args.output_root,
                args.run_root,
                scenario_id=args.scenario_id,
                startup_timeout_s=args.startup_timeout,
                probe_timeout_s=args.probe_timeout,
                flight_timeout_s=args.flight_timeout,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Sandbox flight smoke failed: {error}")
            return 1
        _print({"status": "complete", "run_root": str(root)})
        return 0
    if args.command == "recipe-create":
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
        try:
            result = inspect_recipe(args.input)
        except (FileNotFoundError, TypeError, ValueError) as error:
            print(f"Sandbox recipe failed: {error}")
            return 1
        _print(result)
        return 0
    if args.command == "experiment-run":
        try:
            result = run_recipe(config.project_root, args.recipe)
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
            print(f"Sandbox experiment failed: {error}")
            return 1
        _print(result)
        return 0
    if args.command == "experiment-inspect":
        try:
            result = inspect_result(args.input)
        except (FileNotFoundError, TypeError, ValueError) as error:
            print(f"Sandbox experiment failed: {error}")
            return 1
        _print(result)
        return 0
    if args.command == "supervisor-gate":
        try:
            result = run_supervisor_gate(config.project_root, args.output)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            print(f"Sandbox supervisor gate failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    if args.command == "supervisor-gate-inspect":
        try:
            result = inspect_supervisor_gate(args.input)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            print(f"Sandbox supervisor gate failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    if args.command == "acceptance-run":
        try:
            result = run_acceptance(config.project_root, args.output)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            print(f"Sandbox acceptance failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    if args.command == "acceptance-inspect":
        try:
            result = inspect_acceptance(args.input)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            print(f"Sandbox acceptance failed: {error}")
            return 1
        _print(result)
        return 0 if result["passed"] else 1
    return serve_inspector([
        "--host", args.host,
        "--port", str(args.port),
        "--project-root", str(args.project_root),
    ])
