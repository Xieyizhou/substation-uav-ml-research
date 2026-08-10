"""Identity contract for a deterministic visual-model training view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256


_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _digest(value, name):
    value = str(value)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class TrainingViewIdentity:
    source_development_dataset_identity: str
    sampling_algorithm: str
    sampling_seed: int
    train_membership_sha256: str
    validation_membership_sha256: str
    full_validation_membership_sha256: str
    labels_manifest_sha256: str
    class_order_identity: str
    train_frame_count: int
    validation_frame_count: int
    full_validation_frame_count: int
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]
    train_no_target_count: int
    validation_no_target_count: int
    source_validation_dataset_identity: str | None = None
    identity_schema_version: int = 1

    def __post_init__(self):
        if self.identity_schema_version != 1:
            raise ValueError("unsupported training view identity schema")
        for name in (
            "source_development_dataset_identity",
            "train_membership_sha256",
            "validation_membership_sha256",
            "full_validation_membership_sha256",
            "labels_manifest_sha256",
            "class_order_identity",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.source_validation_dataset_identity is not None:
            object.__setattr__(
                self,
                "source_validation_dataset_identity",
                _digest(
                    self.source_validation_dataset_identity,
                    "source_validation_dataset_identity",
                ),
            )
        if not str(self.sampling_algorithm).strip():
            raise ValueError("sampling_algorithm must not be empty")
        if isinstance(self.sampling_seed, bool) or not isinstance(
            self.sampling_seed, int
        ):
            raise ValueError("sampling_seed must be an integer")
        for name in (
            "train_frame_count",
            "validation_frame_count",
            "full_validation_frame_count",
            "train_no_target_count",
            "validation_no_target_count",
        ):
            _count(getattr(self, name), name)
        for name in ("train_class_counts", "validation_class_counts"):
            counts = dict(getattr(self, name))
            if set(counts) != set(EQUIPMENT_CLASSES):
                raise ValueError(f"{name} must match the locked class order")
            for key, value in counts.items():
                _count(value, f"{name}.{key}")
            object.__setattr__(self, name, dict(sorted(counts.items())))

    def identity_record(self):
        record = asdict(self)
        if self.source_validation_dataset_identity is None:
            record.pop("source_validation_dataset_identity")
        return record

    @property
    def training_view_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {
            **self.identity_record(),
            "training_view_identity_sha256": self.training_view_identity_sha256,
        }

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("training_view_identity_sha256", None)
        identity = cls(**values)
        if supplied != identity.training_view_identity_sha256:
            raise ValueError("training view identity SHA256 mismatch")
        return identity
