"""Identity contracts for licensed real-domain detection datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256


ALLOWED_LICENSES = frozenset({"MIT", "CC0-1.0", "CC-BY-4.0"})
PARTITIONS = ("development", "validation", "blind")
SEMANTIC_DEFINITIONS = {
    "transformer": "complete power transformer equipment",
    "switchgear": "breaker, switchgear, or GIS unit",
    "capacitor_bank": "complete capacitor bank",
    "reactor": "complete shunt or series reactor",
}
MINIMUM_COVERAGE = {
    "development": {"positive_per_class": 400, "no_target": 800},
    "validation": {"positive_per_class": 100, "no_target": 200},
    "blind": {"positive_per_class": 100, "no_target": 200},
}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _digest(value, field):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class RealDomainDatasetIdentity:
    dataset_id: str
    dataset_version: str
    source_registry_sha256: str
    sample_manifest_sha256: str
    managed_manifest_sha256: str
    class_order: tuple[str, ...]
    partition_counts: dict[str, int]
    positive_image_counts: dict[str, dict[str, int]]
    no_target_counts: dict[str, int]
    source_counts: dict[str, int]
    audit_report_sha256: str
    license_ids: tuple[str, ...]
    training_eligible: bool
    identity_schema_version: int = 1

    def __post_init__(self):
        if self.identity_schema_version != 1:
            raise ValueError("unsupported real-domain identity schema")
        if not self.dataset_id.strip() or not self.dataset_version.strip():
            raise ValueError("dataset id and version must not be empty")
        for field in (
            "source_registry_sha256",
            "sample_manifest_sha256",
            "managed_manifest_sha256",
            "audit_report_sha256",
        ):
            _digest(getattr(self, field), field)
        if tuple(self.class_order) != tuple(EQUIPMENT_CLASSES):
            raise ValueError("real-domain class order must match the locked order")
        if set(self.partition_counts) != set(PARTITIONS):
            raise ValueError("real-domain identity must contain every partition")
        if set(self.positive_image_counts) != set(PARTITIONS):
            raise ValueError("positive counts must contain every partition")
        if set(self.no_target_counts) != set(PARTITIONS):
            raise ValueError("no-target counts must contain every partition")
        for partition in PARTITIONS:
            if set(self.positive_image_counts[partition]) != set(EQUIPMENT_CLASSES):
                raise ValueError("positive counts must contain every target class")
        if not set(self.license_ids).issubset(ALLOWED_LICENSES):
            raise ValueError("identity contains an unsupported license")

    def identity_record(self):
        return asdict(self)

    @property
    def real_domain_dataset_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {
            **self.identity_record(),
            "real_domain_dataset_identity_sha256": (
                self.real_domain_dataset_identity_sha256
            ),
        }

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("real_domain_dataset_identity_sha256", None)
        values["class_order"] = tuple(values["class_order"])
        values["license_ids"] = tuple(values["license_ids"])
        identity = cls(**values)
        if supplied != identity.real_domain_dataset_identity_sha256:
            raise ValueError("real-domain dataset identity SHA256 mismatch")
        return identity


def coverage_gaps(positive_counts, no_target_counts):
    gaps = []
    for partition in PARTITIONS:
        required = MINIMUM_COVERAGE[partition]
        for class_name in EQUIPMENT_CLASSES:
            actual = int(positive_counts[partition].get(class_name, 0))
            minimum = required["positive_per_class"]
            if actual < minimum:
                gaps.append(
                    {"partition": partition, "class_name": class_name,
                     "actual": actual, "required": minimum}
                )
        actual = int(no_target_counts.get(partition, 0))
        minimum = required["no_target"]
        if actual < minimum:
            gaps.append(
                {"partition": partition, "class_name": "no_target",
                 "actual": actual, "required": minimum}
            )
    return gaps
