"""CLI registration and dispatch for storage lifecycle operations."""

from __future__ import annotations

from pathlib import Path

from src.sandbox.retention import (
    apply_retention_plan,
    create_retention_plan,
    inspect_retention_plan,
)
from src.sandbox.storage_policy import storage_summary


COMMANDS = frozenset({"storage", "retention-plan", "retention-inspect", "retention-apply"})


def register_storage_commands(commands):
    commands.add_parser("storage", help="Inspect sandbox output use and disk capacity")
    plan = commands.add_parser(
        "retention-plan", help="Preview identity-bound sandbox output pruning"
    )
    plan.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser(
        "retention-inspect", help="Validate a sandbox retention plan"
    )
    inspect.add_argument("--input", type=Path, required=True)
    apply = commands.add_parser(
        "retention-apply", help="Apply a verified sandbox retention plan"
    )
    apply.add_argument("--input", type=Path, required=True)
    apply.add_argument("--confirm-identity", required=True)


def handle_storage_command(args, config):
    if args.command not in COMMANDS:
        return None
    if args.command == "storage":
        return storage_summary(config)
    if args.command == "retention-plan":
        return create_retention_plan(config, args.output)
    plan = inspect_retention_plan(args.input)
    if args.command == "retention-inspect":
        return plan
    if args.confirm_identity != plan["retention_plan_identity_sha256"]:
        raise ValueError("retention confirmation identity mismatch")
    return apply_retention_plan(config, args.input)
