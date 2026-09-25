"""CLI registration and dispatch for storage lifecycle operations."""

from __future__ import annotations

from pathlib import Path
import json

from src.sandbox.retention import (
    apply_retention_plan,
    create_retention_plan,
    inspect_retention_plan,
)
from src.sandbox.storage_policy import storage_summary


COMMANDS = frozenset({"storage", "retention-plan", "retention-inspect", "retention-apply",
                      "camera-archive", "camera-archive-inspect", "camera-archive-restore",
                      "evidence-freeze", "evidence-inspect", "evidence-flight-verify"})


def register_storage_commands(commands):
    check = commands.add_parser("evidence-flight-verify", help="Verify the registered historical SITL snapshot")
    check.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("evidence-freeze", help="Freeze original receipts and their declared inputs")
    freeze.add_argument("--records", type=Path, required=True, help="JSON object mapping record names to paths")
    freeze.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("evidence-inspect", help="Verify a portable frozen evidence archive")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--identity", required=True, help="Trusted manifest SHA-256 returned at freezing")
    for name in ("camera-archive", "camera-archive-inspect", "camera-archive-restore"):
        archive = commands.add_parser(name, help="Manage lossless completed camera evidence archives")
        archive.add_argument("--recording", type=Path, required=True)
        if name == "camera-archive":
            archive.add_argument("--compact", action="store_true",
                                 help="Remove loose raw frames only after archive verification")
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
    if args.command.startswith("evidence-"):
        if args.command == "evidence-flight-verify":
            from src.sandbox.evidence_registry import verify_registered_archive
            return verify_registered_archive(config.project_root, args.output)
        from src.sandbox.evidence_archive import EvidenceArchive, freeze_evidence
        if args.command == "evidence-freeze":
            records = json.loads(args.records.read_text())
            if not isinstance(records, dict):
                raise ValueError("Evidence records must map names to paths")
            return freeze_evidence(records, args.output, project_root=config.project_root)
        with EvidenceArchive(args.input, expected_identity=args.identity) as archive:
            return archive.verify_all()
    if args.command.startswith("camera-archive"):
        if config.profile != "development":
            raise ValueError("camera archival requires development profile")
        root = args.recording.resolve(strict=True)
        if not root.is_relative_to(config.project_root.resolve() / "data/research"):
            raise ValueError("camera archive must stay within project data/research")
        from src.sandbox.camera_archive import (
            archive_camera_recording, inspect_camera_archive, restore_camera_recording,
        )
        if args.command == "camera-archive":
            return archive_camera_recording(root, compact=args.compact)
        if args.command == "camera-archive-restore":
            return restore_camera_recording(root)
        return inspect_camera_archive(root)
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
