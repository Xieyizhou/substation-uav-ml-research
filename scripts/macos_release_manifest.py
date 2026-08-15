#!/usr/bin/env python3
"""Create and verify macOS preview, unsigned Beta, and notarized manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


SCHEMA_VERSION = 4
FORBIDDEN_PARTS = {
    "data",
    "models",
    "outputs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "filename": path.name,
        "kind": path.suffix.removeprefix("."),
        "sha256": sha256(path),
    }


def create_manifest(args: argparse.Namespace) -> None:
    artifacts = [Path(value).resolve() for value in args.artifacts]
    for artifact in artifacts:
        if not artifact.is_file():
            raise SystemExit(f"release artifact is missing: {artifact}")
    payload = {
        "architecture": args.architecture,
        "artifacts": [artifact_record(path) for path in artifacts],
        "distribution_tier": args.distribution_tier,
        "advanced_profiles_require_project_repository": True,
        "advanced_profiles_require_python_environment": True,
        "external_simulator_toolchain": True,
        "macos_app_version": args.version,
        "release_schema_version": SCHEMA_VERSION,
        "notarization": args.notarization,
        "signing": args.signing,
        "source_commit_sha": args.source_commit_sha,
        "standalone_demo_included": True,
        "tracked_worktree_clean": args.tracked_worktree_clean == "true",
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        result[filename.lstrip("* ")] = digest
    return result


def verify_archive_members(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = [Path(name) for name in archive.namelist() if name]
    visible = [member for member in members if member.parts[0] != "__MACOSX"]
    if not visible or any(
        member.parts[0] != "UAV Research Sandbox.app" for member in visible
    ):
        raise SystemExit("ZIP does not contain the expected application bundle")
    for member in visible:
        lowered = {part.lower() for part in member.parts}
        if FORBIDDEN_PARTS.intersection(lowered):
            raise SystemExit(f"forbidden ZIP member: {member}")


def verify_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("release_schema_version") != SCHEMA_VERSION:
        raise SystemExit("unsupported release manifest schema")
    tier = payload.get("distribution_tier")
    if tier not in {"developer_preview", "unsigned_beta", "notarized_beta"}:
        raise SystemExit("unsupported distribution tier")
    if payload.get("standalone_demo_included") is not True:
        raise SystemExit("preview must declare its standalone Demo")
    required_flags = (
        "advanced_profiles_require_project_repository",
        "advanced_profiles_require_python_environment",
        "external_simulator_toolchain",
    )
    if any(payload.get(flag) is not True for flag in required_flags):
        raise SystemExit("preview must declare its external runtime dependencies")
    if tier == "developer_preview" and (
        payload.get("signing"), payload.get("notarization")
    ) != ("ad_hoc", "not_requested"):
        raise SystemExit("preview must use ad-hoc signing without notarization")
    if tier == "unsigned_beta" and (
        payload.get("signing"), payload.get("notarization")
    ) != ("ad_hoc", "not_requested"):
        raise SystemExit("unsigned Beta must use ad-hoc signing without notarization")
    if tier == "notarized_beta" and (
        payload.get("signing"), payload.get("notarization")
    ) != ("developer_id", "stapled"):
        raise SystemExit("notarized Beta must be Developer ID signed and notarized")
    if tier != "developer_preview" and payload.get("tracked_worktree_clean") is not True:
        raise SystemExit("Beta must be built from a clean tracked worktree")
    commit = payload.get("source_commit_sha", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SystemExit("release source commit identity is invalid")

    expected = parse_checksums(Path(args.checksums))
    release_root = manifest_path.parent
    records = payload.get("artifacts", [])
    if len(records) != 2 or {
        record.get("kind") for record in records
    } != {"dmg", "zip"}:
        raise SystemExit("release must contain exactly one DMG and one ZIP")
    prefix = (
        f"UAV-Research-Sandbox-v{payload['macos_app_version']}-macos-"
        f"{payload['architecture']}"
    )
    for record in records:
        filename = record["filename"]
        if filename != f"{prefix}.{record['kind']}":
            raise SystemExit(f"unexpected artifact filename: {filename}")
        if FORBIDDEN_PARTS.intersection(Path(filename).parts):
            raise SystemExit(f"forbidden release path: {filename}")
        path = release_root / filename
        digest = sha256(path)
        if digest != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise SystemExit(f"artifact identity mismatch: {filename}")
        if expected.get(filename) != digest:
            raise SystemExit(f"checksum mismatch: {filename}")
        if record["kind"] == "zip":
            verify_archive_members(path)

    manifest_digest = sha256(manifest_path)
    if expected.get(manifest_path.name) != manifest_digest:
        raise SystemExit("manifest checksum mismatch")
    print(f"Verified macOS {tier} release {payload['macos_app_version']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--version", required=True)
    create.add_argument("--architecture", required=True)
    create.add_argument(
        "--distribution-tier",
        choices=("developer_preview", "unsigned_beta", "notarized_beta"),
        default="developer_preview",
    )
    create.add_argument(
        "--signing", choices=("ad_hoc", "developer_id"), default="ad_hoc"
    )
    create.add_argument(
        "--notarization", choices=("not_requested", "stapled"),
        default="not_requested",
    )
    create.add_argument("--source-commit-sha", required=True)
    create.add_argument(
        "--tracked-worktree-clean", choices=("true", "false"), required=True
    )
    create.add_argument("--output", required=True)
    create.add_argument("artifacts", nargs="+")
    create.set_defaults(handler=create_manifest)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--checksums", required=True)
    verify.set_defaults(handler=verify_manifest)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.handler(parsed)
