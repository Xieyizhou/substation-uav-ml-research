"""Lossless cold storage for completed, non-training RGB sidecar evidence.

Only content-addressed raw camera payloads are compacted. Completion receipts,
model weights, detections and training datasets are never rewritten. Restore
before running legacy tools that require loose payload files.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from src.ml.artifacts import file_sha256, object_sha256, write_json


ARCHIVE = "raw-evidence.zip"
INDEX = "raw-evidence-archive.json"
PAYLOAD = re.compile(r"^payloads/frames/([0-9a-f]{64})\.raw$")


@contextmanager
def _locked(root):
    with (root / ".raw-archive.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("camera archive already in use") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _payload(root, relative):
    if not PAYLOAD.fullmatch(relative):
        raise ValueError("archive accepts only hash-addressed raw camera payloads")
    path = root / relative
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("archive payload cannot traverse a symlink")
    return path


def _completion(root):
    path = root / "completion.json"
    value = json.loads(path.read_text())
    digest = value.pop("identity", None)
    if digest != object_sha256(value):
        raise ValueError("camera completion identity mismatch")
    if (value.get("status") != "live_complete_not_flight_certified"
            or value.get("training_admitted") is not False
            or value.get("control_authority") != "none"):
        raise ValueError("archive requires a completed, non-training, read-only sidecar")
    return value, file_sha256(path)


def _stream_hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _verify_members(archive_path, entries):
    with ZipFile(archive_path) as archive:
        if sorted(archive.namelist()) != sorted(row["path"] for row in entries):
            raise ValueError("archive membership mismatch")
        for row in entries:
            if archive.getinfo(row["path"]).file_size != row["bytes"]:
                raise ValueError("archive payload size mismatch")
            with archive.open(row["path"]) as stream:
                if _stream_hash(stream) != row["sha256"]:
                    raise ValueError("archive payload identity mismatch")


def inspect_camera_archive(recording_root):
    root = Path(recording_root).resolve(strict=True)
    value = json.loads((root / INDEX).read_text())
    supplied = value.pop("archive_identity_sha256", None)
    if supplied != object_sha256(value) or value.get("camera_archive_schema_version") != 1:
        raise ValueError("camera archive index identity mismatch")
    completion, completion_hash = _completion(root)
    if completion_hash != value["completion_sha256"]:
        raise ValueError("camera completion changed after archival")
    for row in value["entries"]:
        path = _payload(root, row["path"])
        if PAYLOAD.fullmatch(row["path"])[1] != row["sha256"]:
            raise ValueError("archive content address mismatch")
        if completion["inputs"].get(str(path)) != row["sha256"]:
            raise ValueError("archive payload is not bound to completion")
    if file_sha256(root / ARCHIVE) != value["archive_sha256"]:
        raise ValueError("camera archive file identity mismatch")
    _verify_members(root / ARCHIVE, value["entries"])
    return {**value, "archive_identity_sha256": supplied}


def archive_camera_recording(recording_root, *, compact=False):
    root = Path(recording_root).resolve(strict=True)
    with _locked(root):
        completion, completion_hash = _completion(root)
        if (root / INDEX).exists():
            record = inspect_camera_archive(root)
        else:
            entries = []
            for path in sorted((root / "payloads/frames").glob("*.raw")):
                relative = path.relative_to(root).as_posix()
                path = _payload(root, relative)
                digest = file_sha256(path)
                if (completion["inputs"].get(str(path)) != digest
                        or PAYLOAD.fullmatch(relative)[1] != digest):
                    raise ValueError("raw payload is not bound to the completed recording")
                stat = path.stat()
                entries.append({"path": relative, "sha256": digest, "bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns, "mode": stat.st_mode & 0o777})
            if not entries:
                raise ValueError("no completed raw camera payloads to archive")
            expected = {path for path in completion["inputs"]
                        if Path(path).parent == root / "payloads/frames"
                        and Path(path).suffix == ".raw"}
            if expected != {str(root / row["path"]) for row in entries}:
                raise ValueError("completed raw payload set is incomplete")
            fd, temporary = tempfile.mkstemp(dir=root, suffix=".zip.partial")
            os.close(fd)
            try:
                with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as archive:
                    for row in entries:
                        archive.write(_payload(root, row["path"]), row["path"])
                _verify_members(temporary, entries)
                if file_sha256(root / "completion.json") != completion_hash:
                    raise ValueError("camera completion changed during archival")
                record = {"camera_archive_schema_version": 1,
                          "completion_sha256": completion_hash, "entries": entries,
                          "archive_sha256": file_sha256(Path(temporary)),
                          "raw_bytes": sum(row["bytes"] for row in entries),
                          "archive_bytes": Path(temporary).stat().st_size}
                record["archive_identity_sha256"] = object_sha256(record)
                # Publishing the verified archive before removing any originals
                # makes interruption recoverable by another call or restore.
                os.replace(temporary, root / ARCHIVE)
                write_json(root / INDEX, record)
            finally:
                Path(temporary).unlink(missing_ok=True)
        removed_bytes = 0
        if compact:
            # Validate the entire loose set before the first removal.
            for row in record["entries"]:
                path = _payload(root, row["path"])
                if path.exists() and file_sha256(path) != row["sha256"]:
                    raise ValueError("loose camera payload changed; refusing compaction")
            if file_sha256(root / "completion.json") != record["completion_sha256"]:
                raise ValueError("camera completion changed before compaction")
            for row in record["entries"]:
                path = _payload(root, row["path"])
                if path.exists():
                    if file_sha256(path) != row["sha256"]:
                        raise ValueError("camera payload changed during compaction")
                    removed_bytes += path.stat().st_size
                    path.unlink()
        return {"archive": str(root / ARCHIVE), "compacted": compact,
                "payload_count": len(record["entries"]), "raw_bytes": record["raw_bytes"],
                "archive_bytes": record["archive_bytes"], "removed_loose_bytes": removed_bytes,
                "archive_identity_sha256": record["archive_identity_sha256"]}


def restore_camera_recording(recording_root):
    root = Path(recording_root).resolve(strict=True)
    with _locked(root):
        record = inspect_camera_archive(root)
        restored = 0
        with ZipFile(root / ARCHIVE) as archive:
            for row in record["entries"]:
                path = _payload(root, row["path"])
                if path.exists():
                    if file_sha256(path) != row["sha256"]:
                        raise ValueError("existing camera payload conflicts with archive")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".restore.partial")
                try:
                    with os.fdopen(fd, "wb") as target, archive.open(row["path"]) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            target.write(block)
                    if file_sha256(Path(temporary)) != row["sha256"]:
                        raise ValueError("restored camera payload identity mismatch")
                    os.chmod(temporary, row["mode"])
                    os.utime(temporary, ns=(row["mtime_ns"], row["mtime_ns"]))
                    os.replace(temporary, path)
                    restored += 1
                finally:
                    Path(temporary).unlink(missing_ok=True)
        return {"restored_count": restored, "payload_count": len(record["entries"]),
                "archive_identity_sha256": record["archive_identity_sha256"]}
