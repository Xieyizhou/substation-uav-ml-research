"""Presentation-safe map studio services."""

from __future__ import annotations

import base64
from pathlib import Path

from src.maps.sandbox_assets import asset_records
from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_preview import preview_record
from src.maps.sandbox_store import SandboxMapStore
from src.maps.sandbox_validation import validate_sandbox_map


def _store(config):
    return SandboxMapStore(config.sandbox_maps_root)


def map_studio_summary(config):
    store = _store(config)
    return {
        "assets": asset_records(),
        "templates": [item.to_record() for item in store.templates()],
        "drafts": [item.to_record() for item in store.drafts()],
        "revisions": list(store.revisions()),
    }


def map_detail(config, map_id):
    store = _store(config)
    try:
        value = store.read_draft(map_id)
        source = "draft"
    except FileNotFoundError:
        value = next((item for item in store.templates() if item.map_id == map_id), None)
        if value is None:
            raise ValueError("sandbox map was not found")
        source = "template"
    report, routes = validate_sandbox_map(value)
    return {
        "source": source,
        "map": value.to_record(),
        "validation": report.to_record(),
        "preview": preview_record(value, routes),
    }


def save_map_draft(config, record):
    if not isinstance(record, dict):
        raise ValueError("sandbox map draft must be an object")
    value = SandboxMap.from_record(record)
    store = _store(config)
    store.save_draft(value)
    return map_detail(config, value.map_id)


def delete_map_draft(config, map_id):
    _store(config).delete_draft(map_id)
    return {"deleted": True, "map_id": map_id}


def create_map_revision(config, map_id):
    root, revision = _store(config).create_revision(map_id)
    return {
        "map_id": map_id,
        "revision": revision.to_record(),
        "path": str(root.relative_to(config.project_root)),
    }


def import_map_bundle(config, encoded):
    if not isinstance(encoded, str):
        raise ValueError("sandbox map bundle is required")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("sandbox map bundle is not valid base64") from error
    value, identity = _store(config).import_bundle(payload)
    return {"map": value.to_record(), "revision": identity}


def map_revision_file(config, map_id, revision_id, name):
    allowed = {"preview.svg", "preview.png", "validation.json", "identity.json"}
    if name not in allowed:
        raise ValueError("sandbox revision file is not exposed")
    return _store(config).revision_file(map_id, revision_id, name)


def map_bundle_file(config, map_id, revision_id):
    payload = _store(config).export_bundle(map_id, revision_id)
    output = config.sandbox_maps_root / "exports" / f"{map_id}-{revision_id[:12]}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return Path(output)
