"""Materialize bounded public previews for quarantine review."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.source_feasibility import load_source_registry


INTAKE_SCHEMA_VERSION = 1
MAX_SAMPLE_COUNT = 16
MAX_SAMPLE_BYTES = 10 * 1024 * 1024
ALLOWED_PREVIEW_HOSTS = frozenset({"source.roboflow.com"})


def intake_real_source(registry_path, source_id, output_root, *, sample_limit=8):
    """Download only registry-pinned public previews into a quarantine folder."""
    _, sources = load_source_registry(registry_path)
    if source_id not in sources:
        raise ValueError(f"unknown real-domain source: {source_id}")
    if not 0 <= sample_limit <= MAX_SAMPLE_COUNT:
        raise ValueError(f"sample_limit must be between 0 and {MAX_SAMPLE_COUNT}")
    source = sources[source_id]
    if source["ingestion_status"] != "quarantine":
        raise ValueError("source intake is limited to quarantine sources")
    root = Path(output_root) / source_id
    _prepare_root(root)
    access = dict(source.get("access", {}))
    urls = list(access.get("review_sample_urls", []))[:sample_limit]
    samples, failures = _download_samples(urls, root / "images")
    receipt = _receipt(source, access, samples, failures)
    write_json(root / "intake_receipt.json", receipt)
    _write_review_queue(root / "review_queue.jsonl", source, samples)
    return receipt


def _prepare_root(root):
    if root.exists():
        raise FileExistsError(f"intake output already exists: {root}")
    (root / "images").mkdir(parents=True)


def _download_samples(urls, image_root):
    samples, failures = [], []
    for index, url in enumerate(urls, start=1):
        try:
            samples.append(_download_one(url, image_root, index))
        except (OSError, ValueError) as exc:
            failures.append({"url": url, "error": str(exc)})
    return samples, failures


def _download_one(url, image_root, index):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_PREVIEW_HOSTS:
        raise ValueError("preview URL host is not allow-listed")
    request = urllib.request.Request(url, headers={"User-Agent": "UAV-Sandbox/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        content_type = response.headers.get_content_type()
        payload = response.read(MAX_SAMPLE_BYTES + 1)
    if content_type not in {"image/jpeg", "image/png"}:
        raise ValueError(f"unexpected preview content type: {content_type}")
    if len(payload) > MAX_SAMPLE_BYTES:
        raise ValueError("preview exceeds size limit")
    suffix = ".png" if content_type == "image/png" else ".jpg"
    path = image_root / f"sample-{index:02d}{suffix}"
    path.write_bytes(payload)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        path.unlink(missing_ok=True)
        raise ValueError(f"invalid preview image: {exc}") from exc
    return {
        "sample_id": f"preview-{index:02d}",
        "relative_path": path.relative_to(image_root.parent).as_posix(),
        "source_url": url,
        "sha256": file_sha256(path),
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
    }


def _receipt(source, access, samples, failures):
    manual = access.get("mode") == "authenticated_manual_export"
    receipt = {
        "real_source_intake_schema_version": INTAKE_SCHEMA_VERSION,
        "source_id": source["source_id"],
        "ingestion_status": source["ingestion_status"],
        "access_mode": access.get("mode", "unavailable"),
        "dataset_locator": access.get("dataset_locator"),
        "requested_export_format": access.get("export_format"),
        "sample_count": len(samples),
        "samples": samples,
        "failures": failures,
        "manual_export_required": manual,
        "training_eligible": False,
        "next_action": (
            "review_public_previews_then_import_authenticated_export"
            if samples else "import_authenticated_export_for_quarantine_review"
        ),
    }
    receipt["intake_identity_sha256"] = object_sha256(receipt)
    return receipt


def _write_review_queue(path, source, samples):
    candidates = sorted(source.get("class_candidates", {}))
    with path.open("w", encoding="utf-8") as destination:
        for sample in samples:
            destination.write(json.dumps({
                "sample_id": sample["sample_id"],
                "relative_path": sample["relative_path"],
                "candidate_classes": candidates,
                "domain_fit": None,
                "whole_equipment_classes": [],
                "source_group_hint": None,
                "review_status": "pending",
            }, sort_keys=True) + "\n")
