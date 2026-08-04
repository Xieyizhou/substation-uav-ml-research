"""Dataset identity contract."""

from dataclasses import asdict, dataclass

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256
from src.vision.contracts.identity_validation import (
    DATASET_ROLES,
    VISUAL_IDENTITY_SCHEMA_VERSION,
    non_negative_int,
    optional_text,
    positive_int,
    required_text,
    sha256,
    stable_tuple,
)


def class_order_identity(classes=EQUIPMENT_CLASSES):
    classes = tuple(classes)
    if classes != tuple(EQUIPMENT_CLASSES):
        raise ValueError("visual class order must match the locked equipment order")
    return object_sha256(
        {"schema_version": VISUAL_IDENTITY_SCHEMA_VERSION, "classes": list(classes)}
    )


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_name: str
    dataset_version: str
    dataset_role: str
    recording_schema_version: int
    annotation_schema_version: int
    decoder_configuration_id: str
    recording_manifest_sha256: str
    scenario_manifest_sha256: str
    split_manifest_sha256: str
    ordered_frame_count: int
    labelled_frame_count: int
    source_recording_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    map_ids: tuple[str, ...]
    seed_ids: tuple[int, ...]
    class_order_identity: str
    annotation_manifest_sha256: str | None = None
    creation_commit_sha: str | None = None
    source_payload_formats: tuple[str, ...] = ("png",)
    canonical_payload_format: str = "png"
    canonical_pixel_format: str = "rgb8"
    identity_schema_version: int = VISUAL_IDENTITY_SCHEMA_VERSION

    def __post_init__(self):
        if self.identity_schema_version != VISUAL_IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dataset identity schema {self.identity_schema_version}"
            )
        object.__setattr__(self, "dataset_name", required_text(self.dataset_name, "dataset_name"))
        object.__setattr__(self, "dataset_version", required_text(self.dataset_version, "dataset_version"))
        if self.dataset_role not in DATASET_ROLES:
            raise ValueError(f"unsupported dataset role: {self.dataset_role!r}")
        positive_int(self.recording_schema_version, "recording_schema_version")
        positive_int(self.annotation_schema_version, "annotation_schema_version")
        for name in (
            "decoder_configuration_id", "recording_manifest_sha256",
            "scenario_manifest_sha256", "split_manifest_sha256",
            "class_order_identity",
        ):
            sha256(getattr(self, name), name)
        non_negative_int(self.ordered_frame_count, "ordered_frame_count")
        non_negative_int(self.labelled_frame_count, "labelled_frame_count")
        if self.labelled_frame_count > self.ordered_frame_count:
            raise ValueError("labelled_frame_count cannot exceed ordered_frame_count")
        annotation_hash = sha256(
            self.annotation_manifest_sha256, "annotation_manifest_sha256", optional=True
        )
        if self.labelled_frame_count and annotation_hash is None:
            raise ValueError("annotation_manifest_sha256 is required for labelled frames")
        object.__setattr__(self, "annotation_manifest_sha256", annotation_hash)
        for name in ("source_recording_ids", "scenario_ids", "map_ids"):
            object.__setattr__(self, name, tuple(sorted(stable_tuple(getattr(self, name), name))))
        seed_ids = tuple(sorted(stable_tuple(self.seed_ids, "seed_ids", cast=int)))
        if any(seed < 0 for seed in seed_ids):
            raise ValueError("seed_ids must be non-negative")
        object.__setattr__(self, "seed_ids", seed_ids)
        formats = tuple(sorted(stable_tuple(self.source_payload_formats, "source_payload_formats")))
        if any(value not in {"png", "jpeg", "raw"} for value in formats):
            raise ValueError("source_payload_formats contains an unsupported format")
        object.__setattr__(self, "source_payload_formats", formats)
        if self.canonical_payload_format != "png":
            raise ValueError("visual identity schema v1 canonical payload must be png")
        if self.canonical_pixel_format != "rgb8":
            raise ValueError("canonical visual pixel format must be rgb8")
        object.__setattr__(self, "creation_commit_sha", optional_text(self.creation_commit_sha, "creation_commit_sha"))

    def identity_record(self):
        return asdict(self)

    @property
    def dataset_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "dataset_identity_sha256": self.dataset_identity_sha256}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("dataset_identity_sha256", None)
        for name in ("source_recording_ids", "scenario_ids", "map_ids", "seed_ids", "source_payload_formats"):
            if name in values:
                values[name] = tuple(values[name])
        identity = cls(**values)
        if supplied != identity.dataset_identity_sha256:
            raise ValueError("dataset identity SHA256 mismatch")
        return identity
