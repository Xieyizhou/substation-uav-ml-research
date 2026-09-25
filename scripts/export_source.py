"""Export the current source tree without Git objects, datasets or environments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile


SOURCE_DIRECTORIES = frozenset({
    ".github", "apps", "archive", "benchmarks", "config", "docs", "scripts",
    "simulation", "src", "tests", "tools",
})
ROOT_FILES = frozenset({
    ".gitignore", ".gitattributes", "README.md", "README.zh-CN.md", "LICENSE", "PROVENANCE.md",
    "THIRD_PARTY_NOTICES.md",
    "PROJECT_STATUS.md", "ROADMAP.md", "CONTRIBUTING.md", "main.py",
    "requirements.txt", "requirements-ml.txt", "requirements-research.txt", "requirements-test.txt",
})
MAX_FILE_BYTES = 100 * 1024 * 1024


def source_paths(root):
    root = Path(root).resolve()
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path: {name}")
        allowed = (
            relative.parts[0] in SOURCE_DIRECTORIES or name in ROOT_FILES
            or name.startswith("data/sample_outputs/")
            or name in {"data/raw_logs/README.md", "outputs/README.md"}
        )
        if not allowed:
            continue
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes project root: {name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Source exceeds 100 MiB: {name}")
        yield name, path


def export_source(root, output):
    output = Path(output).resolve()
    files = list(source_paths(root))
    if not files:
        raise ValueError("No project sources found")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Exclusive creation avoids replacing an existing export.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, path in files:
            payload = path.read_bytes()
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
            entry.external_attr = (stat.S_IFREG | mode) << 16
            bundle.writestr(entry, payload)
            manifest.append({"path": name, "bytes": len(payload),
                             "sha256": hashlib.sha256(payload).hexdigest()})
        entry = zipfile.ZipInfo("SOURCE_MANIFEST.json", date_time=(2020, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(entry, json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "files": len(manifest),
            "source_bytes": sum(row["bytes"] for row in manifest),
            "archive_bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_source(args.project_root, args.output), indent=2))


if __name__ == "__main__":
    main()
