"""Export a selected source tree without datasets, environments or Git objects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sandbox.source_bundle import (
    MANIFEST, PROVENANCE, MAX_FILE_BYTES, ROOT_FILES, SOURCE_DIRECTORIES,
    describe_source, source_paths, source_revision,
)


def export_source(root, output, *, include_untracked=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    files = source_paths(root, include_untracked=include_untracked)
    manifest, provenance = describe_source(files, source_revision(root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for (name, path), row in zip(files, manifest):
            payload = path.read_bytes()
            if (len(payload) != row["bytes"]
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]):
                raise ValueError("Source changed while exporting: " + name)
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
            entry.external_attr = (stat.S_IFREG | mode) << 16
            bundle.writestr(entry, payload)
        for name, value in ((MANIFEST, manifest), (PROVENANCE, provenance)):
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, json.dumps(value, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(output), "files": len(manifest),
        "source_bytes": sum(row["bytes"] for row in manifest),
        "archive_bytes": output.stat().st_size,
        "source_commit_sha": provenance["source_commit_sha"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "includes_untracked": include_untracked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-untracked", action="store_true",
        help="Development-only snapshot; publication defaults to selected tracked files.",
    )
    args = parser.parse_args()
    print(json.dumps(export_source(
        args.project_root, args.output, include_untracked=args.include_untracked,
    ), indent=2))


if __name__ == "__main__":
    main()
