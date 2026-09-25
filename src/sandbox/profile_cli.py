"""CLI registration and dispatch for profile bootstrap and offline gates."""

from __future__ import annotations

from pathlib import Path

from src.sandbox.beta_install import (
    inspect_beta_install_gate,
    run_beta_install_gate,
)
from src.sandbox.bootstrap import bootstrap_sandbox, inspect_bootstrap
from src.sandbox.demo_workflow import inspect_demo, run_demo
from src.sandbox.development_app_gate import (
    inspect_development_app_gate,
    run_development_app_gate,
)
from src.sandbox.release_gate import inspect_release_gate, run_release_gate


COMMANDS = frozenset({
    "bootstrap", "bootstrap-inspect", "demo-run", "demo-inspect",
    "release-gate", "release-gate-inspect", "beta-install-gate",
    "beta-install-gate-inspect",
    "development-app-gate", "development-app-gate-inspect",
})


def register_profile_commands(commands):
    bootstrap = commands.add_parser(
        "bootstrap", help="Initialize the selected local sandbox profile"
    )
    bootstrap.add_argument("--output", type=Path)
    for name, help_text, argument in (
        ("bootstrap-inspect", "Validate a sandbox bootstrap receipt", "input"),
        ("demo-run", "Run the dependency-free demonstration workflow", "output"),
        ("demo-inspect", "Validate a demonstration workflow result", "input"),
        ("release-gate", "Run the offline Sandbox v0.1 release gate", "output"),
        ("release-gate-inspect", "Validate a Sandbox v0.1 gate result", "input"),
        ("beta-install-gate", "Test Demo installation in a clean source and venv", "output"),
        ("beta-install-gate-inspect", "Validate a Beta installation result", "input"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(f"--{argument}", type=Path, required=True)
    development = commands.add_parser(
        "development-app-gate",
        help="Bind one App-managed Development flight to release evidence",
    )
    development.add_argument("--workflow-receipt", type=Path, required=True)
    development.add_argument("--flight-summary", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    development_inspect = commands.add_parser(
        "development-app-gate-inspect",
        help="Validate a Development App flight gate receipt",
    )
    development_inspect.add_argument("--input", type=Path, required=True)
    commands.add_parser("doctor", help="Check dependencies, paths, and disk space")
    commands.add_parser("status", help="Show collection and runtime status")
    challenge = commands.add_parser(
        "challenge-run", help="Run a fresh three-flight LiDAR capability gate"
    )
    challenge.add_argument("--model-id", required=True)


def handle_profile_command(args, config):
    if args.command not in COMMANDS:
        return None
    if args.command == "bootstrap":
        return bootstrap_sandbox(config, args.output), 0
    if args.command == "bootstrap-inspect":
        return inspect_bootstrap(args.input), 0
    if args.command == "demo-run":
        result = run_demo(config.project_root, args.output)
        return result, 0 if result["result"]["passed"] else 1
    inspectors = {
        "demo-inspect": inspect_demo,
        "release-gate-inspect": inspect_release_gate,
        "beta-install-gate-inspect": inspect_beta_install_gate,
        "development-app-gate-inspect": inspect_development_app_gate,
    }
    if args.command in inspectors:
        result = inspectors[args.command](args.input)
    elif args.command == "release-gate":
        result = run_release_gate(config, args.output)
    elif args.command == "beta-install-gate":
        result = run_beta_install_gate(config.project_root, args.output)
    elif args.command == "development-app-gate":
        result = run_development_app_gate(
            config.project_root, args.workflow_receipt,
            args.flight_summary, args.output,
        )
    return result, 0 if result["passed"] else 1
