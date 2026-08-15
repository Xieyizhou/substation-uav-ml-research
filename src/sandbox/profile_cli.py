"""CLI registration and dispatch for profile bootstrap and offline gates."""

from __future__ import annotations

from pathlib import Path

from src.sandbox.beta_install import (
    inspect_beta_install_gate,
    run_beta_install_gate,
)
from src.sandbox.bootstrap import bootstrap_sandbox, inspect_bootstrap
from src.sandbox.demo_workflow import inspect_demo, run_demo
from src.sandbox.release_gate import inspect_release_gate, run_release_gate


COMMANDS = frozenset({
    "bootstrap", "bootstrap-inspect", "demo-run", "demo-inspect",
    "release-gate", "release-gate-inspect", "beta-install-gate",
    "beta-install-gate-inspect",
})


def register_profile_commands(commands):
    bootstrap = commands.add_parser(
        "bootstrap", help="Initialize the selected local sandbox profile"
    )
    bootstrap.add_argument("--output", type=Path)
    bootstrap_inspect = commands.add_parser(
        "bootstrap-inspect", help="Validate a sandbox bootstrap receipt"
    )
    bootstrap_inspect.add_argument("--input", type=Path, required=True)
    demo = commands.add_parser(
        "demo-run", help="Run the dependency-free demonstration workflow"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo_inspect = commands.add_parser(
        "demo-inspect", help="Validate a demonstration workflow result"
    )
    demo_inspect.add_argument("--input", type=Path, required=True)
    release = commands.add_parser(
        "release-gate", help="Run the offline Sandbox v0.1 release gate"
    )
    release.add_argument("--output", type=Path, required=True)
    release_inspect = commands.add_parser(
        "release-gate-inspect", help="Validate a Sandbox v0.1 gate result"
    )
    release_inspect.add_argument("--input", type=Path, required=True)
    beta = commands.add_parser(
        "beta-install-gate", help="Test Demo installation in a clean source and venv"
    )
    beta.add_argument("--output", type=Path, required=True)
    beta_inspect = commands.add_parser(
        "beta-install-gate-inspect", help="Validate a Beta installation result"
    )
    beta_inspect.add_argument("--input", type=Path, required=True)
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
    if args.command == "demo-inspect":
        result = inspect_demo(args.input)
        return result, 0 if result["passed"] else 1
    if args.command == "release-gate":
        result = run_release_gate(config, args.output)
        return result, 0 if result["passed"] else 1
    if args.command == "release-gate-inspect":
        result = inspect_release_gate(args.input)
        return result, 0 if result["passed"] else 1
    if args.command == "beta-install-gate":
        result = run_beta_install_gate(config.project_root, args.output)
        return result, 0 if result["passed"] else 1
    result = inspect_beta_install_gate(args.input)
    return result, 0 if result["passed"] else 1
