"""CLI parser registration for profile bootstrap and offline demo gates."""

from __future__ import annotations

from pathlib import Path


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
    commands.add_parser("doctor", help="Check dependencies, paths, and disk space")
    commands.add_parser("status", help="Show collection and runtime status")
    challenge = commands.add_parser(
        "challenge-run", help="Run a fresh three-flight LiDAR capability gate"
    )
    challenge.add_argument("--model-id", required=True)
