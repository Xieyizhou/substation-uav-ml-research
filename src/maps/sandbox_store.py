"""Safe local storage for sandbox map drafts, revisions, and bundles."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from src.maps.sandbox_contracts import MAP_ID, SandboxMap
from src.maps.sandbox_materialize import materialize_revision
from src.maps.sandbox_templates import sandbox_templates


REVISION_ID = re.compile(r"^[0-9a-f]{64}$")
BUNDLE_FILES = frozenset({
    "map.json", "identity.json", "world.sdf", "obstacles.json",
    "preview.json", "preview.svg", "preview.png", "validation.json",
})


class SandboxMapStore:
    def __init__(self, root):
        self.root = Path(root)

    @property
    def drafts_root(self):
        return self.root / "drafts"

    @property
    def revisions_root(self):
        return self.root / "revisions"

    def _draft_path(self, map_id):
        if not MAP_ID.fullmatch(str(map_id)):
            raise ValueError("invalid sandbox map id")
        return self.drafts_root / str(map_id) / "map.json"

    def _revision_root(self, map_id, revision_id):
        if not MAP_ID.fullmatch(str(map_id)) or not REVISION_ID.fullmatch(str(revision_id)):
            raise ValueError("invalid sandbox map revision")
        return self.revisions_root / str(map_id) / str(revision_id)

    def save_draft(self, map_value: SandboxMap):
        path = self._draft_path(map_value.map_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(map_value.to_record(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return map_value

    def read_draft(self, map_id):
        return SandboxMap.from_record(json.loads(self._draft_path(map_id).read_text()))

    def delete_draft(self, map_id):
        path = self._draft_path(map_id)
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass

    def drafts(self):
        values = []
        for path in sorted(self.drafts_root.glob("*/map.json")):
            try:
                values.append(SandboxMap.from_record(json.loads(path.read_text())))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return tuple(values)

    def templates(self):
        return sandbox_templates()

    def revisions(self, map_id=None):
        pattern = f"{map_id}/*/identity.json" if map_id else "*/*/identity.json"
        values = []
        for path in sorted(self.revisions_root.glob(pattern), reverse=True):
            try:
                values.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return tuple(values)

    def create_revision(self, map_id):
        return materialize_revision(self.read_draft(map_id), self.revisions_root)

    def revision_file(self, map_id, revision_id, relative):
        root = self._revision_root(map_id, revision_id).resolve()
        candidate = (root / relative).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise ValueError("sandbox revision file is unavailable")
        return candidate

    def revision_root(self, map_id, revision_id):
        root = self._revision_root(map_id, revision_id)
        identity = json.loads((root / "identity.json").read_text())
        if identity.get("revision_identity_sha256") != revision_id:
            raise ValueError("sandbox map revision identity mismatch")
        return root

    def accepted_route_artifacts(self, map_id, revision_id, mission_id):
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", str(mission_id)):
            raise ValueError("invalid sandbox mission id")
        root = self.revision_root(map_id, revision_id)
        identity = json.loads((root / "identity.json").read_text())
        relative_paths = (
            "map.json", "validation.json", f"routes/{mission_id}.json",
            f"routes/{mission_id}.quality.json",
        )
        records = {}
        expected = identity.get("artifact_sha256", {})
        for relative in relative_paths:
            path = self.revision_file(map_id, revision_id, relative)
            payload = path.read_bytes()
            if expected.get(relative) != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"sandbox revision artifact hash mismatch: {relative}")
            records[relative] = json.loads(payload)
        validation = records["validation.json"]
        route = records[f"routes/{mission_id}.json"]
        quality = records[f"routes/{mission_id}.quality.json"]
        if validation.get("valid") is not True or quality.get("accepted") is not True:
            raise ValueError("sandbox route has not passed the route quality gate")
        if quality.get("route_identity_sha256") != route.get("route_identity_sha256"):
            raise ValueError("sandbox route quality identity binding mismatch")
        quality_identity = quality.get("route_quality_identity_sha256")
        if quality_identity not in validation.get("route_quality_identities", []):
            raise ValueError("sandbox route quality receipt is not in validation")
        return root, records["map.json"], route, quality

    def export_bundle(self, map_id, revision_id):
        root = self._revision_root(map_id, revision_id)
        identity = json.loads((root / "identity.json").read_text())
        if identity.get("revision_identity_sha256") != revision_id:
            raise ValueError("sandbox revision identity mismatch")
        output = BytesIO()
        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
        return output.getvalue()

    def import_bundle(self, payload):
        if len(payload) > 2_000_000:
            raise ValueError("sandbox map bundle exceeds 2 MB")
        with ZipFile(BytesIO(payload)) as archive:
            names = set(archive.namelist())
            if not BUNDLE_FILES.issubset(names) or any(
                name.startswith("/") or ".." in Path(name).parts for name in names
            ):
                raise ValueError("sandbox map bundle layout is invalid")
            map_value = SandboxMap.from_record(json.loads(archive.read("map.json")))
            identity = json.loads(archive.read("identity.json"))
            revision_id = identity.get("revision_identity_sha256")
            root = self._revision_root(map_value.map_id, revision_id)
            if root.exists():
                return map_value, identity
            expected = identity.get("artifact_sha256", {})
            for relative, digest in expected.items():
                if relative not in names or hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                    raise ValueError(f"sandbox map bundle hash mismatch: {relative}")
            root.mkdir(parents=True)
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
        self.save_draft(map_value)
        return map_value, identity
