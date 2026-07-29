"""Versioned, deterministic identities for visual research artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePath
import re

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256


VISUAL_IDENTITY_SCHEMA_VERSION = 1
DATASET_ROLES = frozenset(
    {
        "pilot",
        "development",
        "validation",
        "held_out_test",
        "formal",
        "external_real_image",
    }
)
PRECISIONS = frozenset({"fp32", "fp16", "int8"})
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def _required_text(value, name):
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    if (
        PurePath(value).is_absolute()
        or value.startswith("~/")
        or _WINDOWS_ABSOLUTE_PATTERN.match(value)
    ):
        raise ValueError(f"{name} must not contain an absolute path")
    return value


def _optional_text(value, name):
    return None if value is None else _required_text(value, name)


def _sha256(value, name, *, optional=False):
    if value is None and optional:
        return None
    value = str(value)
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _stable_tuple(values, name, *, cast=str, allow_empty=False):
    result = tuple(cast(value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if cast is str:
            _required_text(value, name)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def class_order_identity(classes=EQUIPMENT_CLASSES):
    classes = tuple(classes)
    if classes != tuple(EQUIPMENT_CLASSES):
        raise ValueError("visual class order must match the locked equipment order")
    return object_sha256(
        {
            "schema_version": VISUAL_IDENTITY_SCHEMA_VERSION,
            "classes": list(classes),
        }
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
        object.__setattr__(
            self, "dataset_name", _required_text(self.dataset_name, "dataset_name")
        )
        object.__setattr__(
            self,
            "dataset_version",
            _required_text(self.dataset_version, "dataset_version"),
        )
        if self.dataset_role not in DATASET_ROLES:
            raise ValueError(f"unsupported dataset role: {self.dataset_role!r}")
        _positive_int(self.recording_schema_version, "recording_schema_version")
        _positive_int(self.annotation_schema_version, "annotation_schema_version")
        for name in (
            "decoder_configuration_id",
            "recording_manifest_sha256",
            "scenario_manifest_sha256",
            "split_manifest_sha256",
            "class_order_identity",
        ):
            _sha256(getattr(self, name), name)
        _non_negative_int(self.ordered_frame_count, "ordered_frame_count")
        _non_negative_int(self.labelled_frame_count, "labelled_frame_count")
        if self.labelled_frame_count > self.ordered_frame_count:
            raise ValueError("labelled_frame_count cannot exceed ordered_frame_count")
        annotation_hash = _sha256(
            self.annotation_manifest_sha256,
            "annotation_manifest_sha256",
            optional=True,
        )
        if self.labelled_frame_count and annotation_hash is None:
            raise ValueError(
                "annotation_manifest_sha256 is required for labelled frames"
            )
        object.__setattr__(self, "annotation_manifest_sha256", annotation_hash)
        for name in ("source_recording_ids", "scenario_ids", "map_ids"):
            object.__setattr__(
                self, name, tuple(sorted(_stable_tuple(getattr(self, name), name)))
            )
        seed_ids = tuple(
            sorted(_stable_tuple(self.seed_ids, "seed_ids", cast=int))
        )
        if any(seed < 0 for seed in seed_ids):
            raise ValueError("seed_ids must be non-negative")
        object.__setattr__(self, "seed_ids", seed_ids)
        formats = tuple(
            sorted(
                _stable_tuple(
                    self.source_payload_formats, "source_payload_formats"
                )
            )
        )
        if any(value not in {"png", "jpeg", "raw"} for value in formats):
            raise ValueError("source_payload_formats contains an unsupported format")
        object.__setattr__(self, "source_payload_formats", formats)
        if self.canonical_payload_format != "png":
            raise ValueError("visual identity schema v1 canonical payload must be png")
        if self.canonical_pixel_format != "rgb8":
            raise ValueError("canonical visual pixel format must be rgb8")
        object.__setattr__(
            self,
            "creation_commit_sha",
            _optional_text(self.creation_commit_sha, "creation_commit_sha"),
        )

    def identity_record(self):
        return asdict(self)

    @property
    def dataset_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {
            **self.identity_record(),
            "dataset_identity_sha256": self.dataset_identity_sha256,
        }

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("dataset_identity_sha256", None)
        for name in (
            "source_recording_ids",
            "scenario_ids",
            "map_ids",
            "seed_ids",
            "source_payload_formats",
        ):
            if name in values:
                values[name] = tuple(values[name])
        identity = cls(**values)
        if supplied != identity.dataset_identity_sha256:
            raise ValueError("dataset identity SHA256 mismatch")
        return identity


@dataclass(frozen=True)
class PreprocessingIdentity:
    target_input_width: int
    target_input_height: int
    resize_policy: str
    letterbox_policy: str
    interpolation_method: str
    padding_value: tuple[int, int, int]
    hwc_to_chw: bool
    rgb_bgr_policy: str
    uint8_to_float: bool
    normalization_scale: float
    mean: tuple[float, float, float] | None
    standard_deviation: tuple[float, float, float] | None
    batch_dimension_policy: str
    tensor_dtype: str
    contiguous_memory_policy: str
    implementation: str
    implementation_version: str
    identity_schema_version: int = VISUAL_IDENTITY_SCHEMA_VERSION

    def __post_init__(self):
        if self.identity_schema_version != VISUAL_IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported preprocessing identity schema "
                f"{self.identity_schema_version}"
            )
        _positive_int(self.target_input_width, "target_input_width")
        _positive_int(self.target_input_height, "target_input_height")
        for name in (
            "resize_policy",
            "letterbox_policy",
            "interpolation_method",
            "rgb_bgr_policy",
            "batch_dimension_policy",
            "tensor_dtype",
            "contiguous_memory_policy",
            "implementation",
            "implementation_version",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        padding = tuple(self.padding_value)
        if len(padding) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 255
            for value in padding
        ):
            raise ValueError("padding_value must contain three uint8 values")
        object.__setattr__(self, "padding_value", padding)
        if not isinstance(self.hwc_to_chw, bool) or not isinstance(
            self.uint8_to_float, bool
        ):
            raise ValueError("preprocessing conversion policies must be boolean")
        if self.normalization_scale <= 0:
            raise ValueError("normalization_scale must be positive")
        object.__setattr__(
            self, "normalization_scale", float(self.normalization_scale)
        )
        for name in ("mean", "standard_deviation"):
            value = getattr(self, name)
            if value is not None:
                value = tuple(float(item) for item in value)
                if len(value) != 3:
                    raise ValueError(f"{name} must contain three values")
                if name == "standard_deviation" and any(item <= 0 for item in value):
                    raise ValueError("standard_deviation values must be positive")
                object.__setattr__(self, name, value)

    def identity_record(self):
        return asdict(self)

    @property
    def preprocessing_configuration_id(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {
            **self.identity_record(),
            "preprocessing_configuration_id": self.preprocessing_configuration_id,
        }

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("preprocessing_configuration_id", None)
        for name in ("padding_value", "mean", "standard_deviation"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        identity = cls(**values)
        if supplied != identity.preprocessing_configuration_id:
            raise ValueError("preprocessing configuration SHA256 mismatch")
        return identity


@dataclass(frozen=True)
class ModelIdentity:
    model_family: str
    architecture_variant: str
    task: str
    class_order_identity: str
    input_width: int
    input_height: int
    input_pixel_format: str
    preprocessing_configuration_id: str
    runtime_backend: str
    precision: str
    weights_sha256: str
    model_file_sha256: str
    model_package_schema_version: int
    runtime_version_identity: str
    training_dataset_identity: str | None = None
    training_code_commit_sha: str | None = None
    export_configuration_identity: str | None = None
    onnx_opset: int | None = None
    identity_schema_version: int = VISUAL_IDENTITY_SCHEMA_VERSION

    def __post_init__(self):
        if self.identity_schema_version != VISUAL_IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model identity schema {self.identity_schema_version}"
            )
        for name in (
            "model_family",
            "architecture_variant",
            "task",
            "runtime_backend",
            "runtime_version_identity",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        _positive_int(self.input_width, "input_width")
        _positive_int(self.input_height, "input_height")
        _positive_int(
            self.model_package_schema_version, "model_package_schema_version"
        )
        if self.input_pixel_format != "rgb8":
            raise ValueError("visual model input_pixel_format must be rgb8")
        for name in (
            "class_order_identity",
            "preprocessing_configuration_id",
            "weights_sha256",
            "model_file_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in (
            "training_dataset_identity",
            "export_configuration_identity",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name, optional=True)
            )
        object.__setattr__(
            self,
            "training_code_commit_sha",
            _optional_text(
                self.training_code_commit_sha, "training_code_commit_sha"
            ),
        )
        if self.precision not in PRECISIONS and not self.precision.startswith(
            "other:"
        ):
            raise ValueError("precision must be fp32, fp16, int8, or other:<name>")
        if self.precision.startswith("other:") and not self.precision[6:].strip():
            raise ValueError("other precision must be explicitly named")
        if self.onnx_opset is not None:
            _positive_int(self.onnx_opset, "onnx_opset")

    def identity_record(self):
        return asdict(self)

    @property
    def model_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {
            **self.identity_record(),
            "model_identity_sha256": self.model_identity_sha256,
        }

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("model_identity_sha256", None)
        identity = cls(**values)
        if supplied != identity.model_identity_sha256:
            raise ValueError("model identity SHA256 mismatch")
        return identity
