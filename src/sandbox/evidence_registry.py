"""Registered historical evidence and explicit, repeatable verification jobs."""

from datetime import datetime, timezone
import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256
from src.sandbox.frozen_flight_evidence import verify_flight_archive

REGISTRY = Path("config/sandbox/flight_evidence_archive.json")
CHECKS = Path("outputs/sandbox/evidence/checks")


def registered_archive(root):
    root = Path(root).resolve()
    value = json.loads((root / REGISTRY).read_text())
    relative = Path(value["path"])
    if value.get("schema_version") != 1 or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid evidence archive registration")
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError("Evidence archive escapes project root")
    return value, path


def verify_registered_archive(root, output):
    entry, archive = registered_archive(root)
    before = archive.stat()
    result = verify_flight_archive(archive, entry["manifest_sha256"])
    digest = file_sha256(archive)
    after = archive.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Evidence archive changed during verification")
    result.update(verified_at=datetime.now(timezone.utc).isoformat(),
                  archive_path=entry["path"], archive_sha256=digest,
                  archive_bytes=after.st_size, archive_mtime_ns=after.st_mtime_ns)
    result["verification_identity_sha256"] = object_sha256(result)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def evidence_summary(root):
    root = Path(root)
    result = dict(available=False, last_verification=None, current_runtime_certified=False,
                  reason="尚未安装历史证据包。源码可以独立使用；历史飞行复核需要对应证据包。")
    try:
        entry, archive = registered_archive(root)
        stat = archive.stat()
        result.update(available=True, archive_bytes=stat.st_size,
                      manifest_sha256=entry["manifest_sha256"], reason="证据包已就绪，可重新核验。")
        for path in sorted((root / CHECKS).glob("*.json"), reverse=True):
            try:
                report = json.loads(path.read_text())
                identity = report.pop("verification_identity_sha256")
                if identity != object_sha256(report) or report["manifest_sha256"] != entry["manifest_sha256"]:
                    continue
                result["last_verification"] = report
                result["changed_since_verification"] = (report["archive_bytes"], report["archive_mtime_ns"]) != (stat.st_size, stat.st_mtime_ns)
                result["reason"] = "显示上次核验结果；重新核验会再次读取并校验完整证据包。"
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["detail"] = str(error)
    return result
