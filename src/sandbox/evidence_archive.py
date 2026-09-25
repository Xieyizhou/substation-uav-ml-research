"""Content-addressed evidence snapshots; original receipts are never rewritten.

Original paths are opaque provenance keys, never filesystem destinations when
reading an archive. Verification covers explicitly declared records and all
their direct inputs. A JSON input is not implicitly a separately audited record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

from src.ml.artifacts import file_sha256, object_sha256

SCHEMA = "sandbox-evidence-archive-v1"
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
CHUNK = 1024 * 1024


def _digest(value):
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise ValueError("Invalid evidence SHA-256")
    return value


def _inputs(record):
    value = record.get("inputs") if isinstance(record, dict) else None
    if not isinstance(value, dict) or not value:
        raise ValueError("Evidence record must declare nonempty inputs")
    for path, digest in value.items():
        if not isinstance(path, str) or not path:
            raise ValueError("Invalid evidence provenance key")
        _digest(digest)
    return value


def freeze_evidence(records, output, *, project_root, max_bytes=32 * 1024**3):
    """Snapshot named records and their exact declared inputs, exclusively.

    max_bytes bounds uncompressed content. Copy hashing detects changes during
    freezing; conflicting revisions of a path require separate archives.
    """
    root, output = Path(project_root).resolve(), Path(output)
    paths, expected, named = {}, {}, {}

    def include(key, digest):
        if key in expected and expected[key] != digest:
            raise ValueError("Conflicting evidence revisions: " + key)
        source = Path(key)
        source = source if source.is_absolute() else root / source
        if source.is_symlink() or not source.is_file():
            raise ValueError("Missing or symbolic evidence file: " + key)
        paths[key], expected[key] = source, _digest(digest)

    if not records:
        raise ValueError("At least one evidence record is required")
    for name, path in records.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid evidence record name")
        path = Path(path)
        path = path if path.is_absolute() else root / path
        key = str(path.absolute())
        payload = path.read_bytes()
        include(key, hashlib.sha256(payload).hexdigest())
        named[name] = key
        for source, digest in _inputs(json.loads(payload)).items():
            include(source, digest)
    sizes = {key: path.stat().st_size for key, path in paths.items()}
    if sum(sizes.values()) > max_bytes:
        raise ValueError("Evidence snapshot exceeds byte budget")
    manifest = dict(schema=SCHEMA, scope="declared_records_and_direct_inputs",
                    records=named, files={key: dict(sha256=expected[key], bytes=sizes[key])
                                         for key in sorted(paths)})
    identity = object_sha256(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("xb") as destination:
            created = True
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=3, allowZip64=True) as archive:
                written = set()
                for key, path in paths.items():
                    digest = expected[key]
                    if digest in written:
                        if file_sha256(path) != digest:
                            raise ValueError("Stale evidence: " + key)
                        continue
                    actual, count = hashlib.sha256(), 0
                    with path.open("rb") as source, archive.open("objects/" + digest, "w", force_zip64=True) as target:
                        for block in iter(lambda: source.read(CHUNK), b""):
                            count += len(block)
                            if count > sizes[key]:
                                raise ValueError("Evidence grew while freezing: " + key)
                            actual.update(block)
                            target.write(block)
                    if actual.hexdigest() != digest or count != sizes[key]:
                        raise ValueError("Stale evidence: " + key)
                    written.add(digest)
                archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    return dict(path=str(output), manifest_sha256=identity, records=len(named),
                files=len(paths), objects=len(written), source_bytes=sum(sizes.values()),
                archive_bytes=output.stat().st_size)


class EvidenceArchive:
    """Read a snapshot using an identity pinned outside the archive."""

    def __init__(self, path, *, expected_identity, max_bytes=32 * 1024**3):
        self.archive = zipfile.ZipFile(path)
        try:
            infos = self.archive.infolist()
            if len({item.filename for item in infos}) != len(infos):
                raise ValueError("Duplicate evidence archive member")
            if self.archive.getinfo("manifest.json").file_size > 32 * 1024**2:
                raise ValueError("Evidence manifest is too large")
            self.manifest = json.loads(self.archive.read("manifest.json"))
            if object_sha256(self.manifest) != _digest(expected_identity):
                raise ValueError("Evidence manifest identity changed")
            if self.manifest.get("schema") != SCHEMA or self.manifest.get("scope") != "declared_records_and_direct_inputs":
                raise ValueError("Unsupported evidence archive schema")
            self.files, self.records = self.manifest["files"], self.manifest["records"]
            if not self.records or not self.files:
                raise ValueError("Empty evidence archive")
            objects = {}
            for entry in self.files.values():
                digest, size = _digest(entry["sha256"]), entry["bytes"]
                if type(size) is not int or size < 0:
                    raise ValueError("Invalid evidence size")
                if digest in objects and objects[digest] != size:
                    raise ValueError("Conflicting evidence object sizes")
                objects[digest] = size
            if sum(objects.values()) > max_bytes:
                raise ValueError("Evidence archive exceeds byte budget")
            if set(self.records.values()) - set(self.files):
                raise ValueError("Missing declared evidence record")
            if {i.filename for i in infos} != {"manifest.json", *("objects/" + d for d in objects)}:
                raise ValueError("Missing or unexpected evidence member")
            for digest, size in objects.items():
                info = self.archive.getinfo("objects/" + digest)
                if info.file_size != size or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Invalid evidence archive member")
            self.objects = objects
            self.identity = expected_identity
        except BaseException:
            self.archive.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()

    def verify_all(self):
        for digest in self.objects:
            actual = hashlib.sha256()
            with self.archive.open("objects/" + digest) as source:
                for block in iter(lambda: source.read(CHUNK), b""):
                    actual.update(block)
            if actual.hexdigest() != digest:
                raise ValueError("Evidence object changed: " + digest)
        for name in self.records:
            self.record(name)
        return dict(manifest_sha256=self.identity, records=len(self.records),
                    objects=len(self.objects), source_bytes=sum(self.objects.values()))

    def read(self, key, *, max_bytes=32 * 1024**2):
        entry = self.files[key]
        if entry["bytes"] > max_bytes:
            raise ValueError("Evidence read exceeds byte budget")
        payload = self.archive.read("objects/" + entry["sha256"])
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Evidence object changed: " + key)
        return payload

    def record(self, name):
        value = json.loads(self.read(self.records[name]))
        for key, digest in _inputs(value).items():
            if key not in self.files or self.files[key]["sha256"] != digest:
                raise ValueError("Evidence dependency missing or changed: " + key)
        return value
