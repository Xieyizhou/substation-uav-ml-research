"""One source inventory for exports, clean installs and unpacked bundles."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from src.ml.artifacts import file_sha256, object_sha256

SOURCE_DIRECTORIES = frozenset({
    ".github", "apps", "archive", "benchmarks", "config", "docs", "examples",
    "scripts", "simulation", "src", "tests", "tools",
})
ROOT_FILES = frozenset({
    ".gitignore", ".gitattributes", "README.md", "README.zh-CN.md", "LICENSE",
    "PROVENANCE.md", "THIRD_PARTY_NOTICES.md", "PROJECT_STATUS.md", "ROADMAP.md",
    "CONTRIBUTING.md", "main.py", "requirements.txt", "requirements-offline.txt",
    "requirements-ml.txt", "requirements-research.txt", "requirements-test.txt",
    "pyproject.toml", "pytest.ini",
})
MAX_FILE_BYTES = 100 * 1024 * 1024
MANIFEST = "SOURCE_MANIFEST.json"
PROVENANCE = "SOURCE_PROVENANCE.json"


def allowed_path(name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or relative.as_posix() != name):
        return False
    return (relative.parts[0] in SOURCE_DIRECTORIES or name in ROOT_FILES
            or name.startswith("data/sample_outputs/")
            or name in {"data/raw_logs/README.md", "outputs/README.md"})


def _bundle_records(root):
    records = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    provenance = json.loads((root / PROVENANCE).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Source manifest must contain files")
    if (not isinstance(provenance, dict)
            or provenance.get("source_bundle_schema_version") != 1
            or provenance.get("source_manifest_sha256") != object_sha256(records)):
        raise ValueError("Source manifest identity differs from provenance")
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Source manifest rows must be objects")
        name = row.get("path")
        if not isinstance(name, str) or not allowed_path(name) or name in seen:
            raise ValueError("Unsafe or duplicate source manifest path")
        seen.add(name)
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root)
                or path.stat().st_size != row.get("bytes")
                or path.stat().st_size > MAX_FILE_BYTES
                or file_sha256(path) != row.get("sha256")):
            raise ValueError("Source bundle content changed: " + name)
    if [row["path"] for row in records] != sorted(seen):
        raise ValueError("Source manifest must be ordered")
    return records, provenance


def source_paths(root, *, include_untracked=False):
    root = Path(root).resolve()
    if not (root / ".git").exists():
        if include_untracked:
            raise ValueError("Development working-tree export requires Git")
        if not (root / MANIFEST).is_file() or not (root / PROVENANCE).is_file():
            raise ValueError("Source tree requires its own Git metadata or source bundle manifests")
        records, _ = _bundle_records(root)
        return [(row["path"], root / row["path"]) for row in records]
    command = ["git", "ls-files", "--cached", "-z"]
    if include_untracked:
        command += ["--others", "--exclude-standard"]
    result = subprocess.run(command, cwd=root, check=True, capture_output=True, timeout=30)
    paths = []
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe source path: " + name)
        if not allowed_path(name):
            continue
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError("Source escapes project root: " + name)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Source exceeds 100 MiB: " + name)
        paths.append((name, path))
    if not paths:
        raise ValueError("No selected source files found")
    return paths


def source_revision(root):
    root = Path(root).resolve()
    if not (root / ".git").exists():
        try:
            # A recorded commit describes this tree only while its bytes match.
            # Receipt callers do not necessarily call source_paths first.
            _, provenance = _bundle_records(root)
            revision = provenance["source_commit_sha"]
            if re.fullmatch(r"[0-9a-f]{40}(?:-dirty)?|unknown", revision):
                return revision
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return "unknown"
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=3,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True, timeout=3,
            stderr=subprocess.DEVNULL,
        ).strip()
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def describe_source(files, revision):
    records = [
        dict(path=name, bytes=path.stat().st_size, sha256=file_sha256(path))
        for name, path in files
    ]
    provenance = dict(
        source_bundle_schema_version=1, source_commit_sha=revision,
        source_manifest_sha256=object_sha256(records),
    )
    return records, provenance


def write_source_metadata(destination, records, provenance):
    destination = Path(destination)
    for name, value in ((MANIFEST, records), (PROVENANCE, provenance)):
        (destination / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
