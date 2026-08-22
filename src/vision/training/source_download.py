"""Download one registry-pinned, provenance-approved public source artifact."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from src.ml.artifacts import object_sha256, write_json
from src.vision.training.source_feasibility import (
    load_source_registry,
    source_is_training_eligible,
)


DOWNLOAD_SCHEMA_VERSION = 1
ALLOWED_DOWNLOAD_HOSTS = frozenset({"zenodo.org"})
MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024 * 1024


def download_approved_source(registry_path, source_id, output_root):
    """Fetch a fixed public artifact and verify its registry-pinned checksum."""
    _, sources = load_source_registry(registry_path)
    if source_id not in sources:
        raise ValueError(f"unknown real-domain source: {source_id}")
    source = sources[source_id]
    if not source_is_training_eligible(source):
        raise ValueError("automatic download is limited to approved sources")
    artifact = dict(source.get("public_artifact", {}))
    _validate_artifact(artifact)
    root = Path(output_root) / source_id
    root.mkdir(parents=True, exist_ok=True)
    destination = root / artifact["filename"]
    receipt_path = root / "download_receipt.json"
    if destination.exists() or receipt_path.exists():
        raise FileExistsError(f"source download output already exists: {root}")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"partial source download already exists: {temporary}")
    try:
        result = _stream_download(artifact, temporary)
        temporary.replace(destination)
    except BaseException:
        # Preserve a partial file for diagnosis; never report it as complete.
        raise
    receipt = {
        "real_source_download_schema_version": DOWNLOAD_SCHEMA_VERSION,
        "source_id": source_id,
        "filename": destination.name,
        "bytes": result["bytes"],
        "sha256": result["sha256"],
        "upstream_checksum": artifact["upstream_checksum"],
        "source_url": artifact["url"],
        "training_eligible": False,
        "next_action": "convert_annotations_and_create_canonical_manifest",
    }
    receipt["download_identity_sha256"] = object_sha256(receipt)
    write_json(receipt_path, receipt)
    return receipt


def _validate_artifact(artifact):
    required = {"url", "filename", "bytes", "upstream_checksum"}
    if not required.issubset(artifact):
        raise ValueError("approved source has no complete public artifact record")
    parsed = urlparse(str(artifact["url"]))
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError("public artifact host is not allow-listed")
    filename = str(artifact["filename"])
    if Path(filename).name != filename or not filename:
        raise ValueError("public artifact filename is invalid")
    expected_bytes = int(artifact["bytes"])
    if not 0 < expected_bytes <= MAX_DOWNLOAD_BYTES:
        raise ValueError("public artifact size is outside the allowed range")
    algorithm, _, digest = str(artifact["upstream_checksum"]).partition(":")
    if algorithm != "md5" or len(digest) != 32:
        raise ValueError("public artifact requires a valid upstream MD5")


def _stream_download(artifact, temporary):
    request = urllib.request.Request(
        artifact["url"], headers={"User-Agent": "UAV-Sandbox/1"}
    )
    expected_bytes = int(artifact["bytes"])
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        with temporary.open("xb") as destination:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > expected_bytes or written > MAX_DOWNLOAD_BYTES:
                    raise ValueError("download exceeds the registry-pinned size")
                destination.write(block)
                md5.update(block)
                sha256.update(block)
    if written != expected_bytes:
        raise ValueError("download size does not match the registry")
    expected_md5 = artifact["upstream_checksum"].split(":", 1)[1]
    if md5.hexdigest() != expected_md5:
        raise ValueError("download checksum does not match the registry")
    return {"bytes": written, "sha256": sha256.hexdigest()}
