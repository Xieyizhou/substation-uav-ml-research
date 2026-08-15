"""Validated version identities shared by the service and release tooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


@dataclass(frozen=True)
class SandboxVersion:
    sandbox_product_version: str
    macos_app_version: str
    operator_api_version: str
    gate_schema_version: int

    def __post_init__(self):
        for name in (
            "sandbox_product_version", "macos_app_version", "operator_api_version",
        ):
            if not VERSION_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"invalid {name}")
        if self.gate_schema_version < 1:
            raise ValueError("invalid gate_schema_version")

    def to_record(self):
        return asdict(self)


def load_sandbox_version(project_root) -> SandboxVersion:
    path = Path(project_root) / "config/sandbox/version.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sandbox version manifest must be an object")
    return SandboxVersion(**value)
