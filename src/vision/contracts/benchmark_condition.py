"""Static visual benchmark condition contract."""

from dataclasses import asdict, dataclass
import math

from src.ml.artifacts import object_sha256
from src.vision.contracts.identity import DatasetIdentity, ModelIdentity, PreprocessingIdentity
from src.vision.contracts.identity_validation import (
    non_negative_int,
    positive_int,
    required_text,
    sha256,
)


VISUAL_BENCHMARK_CONDITION_SCHEMA_VERSION = 1
INFERENCE_POLICIES = frozenset({"every_frame", "every_nth_frame"})


def finite_non_negative(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


@dataclass(frozen=True)
class VisualBenchmarkCondition:
    condition_id: str
    dataset_identity_sha256: str
    decoder_configuration_id: str
    preprocessing_configuration_id: str
    model_identity_sha256: str
    input_width: int
    input_height: int
    inference_policy: str
    frame_skip_interval: int
    confidence_threshold: float
    target_inference_rate_hz: float | None
    roi_mode: str
    batch_size: int
    warmup_frame_count: int
    measured_frame_count: int
    runtime_backend: str
    device_identity: str
    precision: str
    deadline_definition: str
    random_seed: int | None
    software_commit_sha: str
    condition_schema_version: int = VISUAL_BENCHMARK_CONDITION_SCHEMA_VERSION

    def __post_init__(self):
        if self.condition_schema_version != VISUAL_BENCHMARK_CONDITION_SCHEMA_VERSION:
            raise ValueError(f"unsupported visual condition schema {self.condition_schema_version}")
        for name in (
            "condition_id", "runtime_backend", "device_identity", "precision",
            "deadline_definition", "software_commit_sha",
        ):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        for name in (
            "dataset_identity_sha256", "decoder_configuration_id",
            "preprocessing_configuration_id", "model_identity_sha256",
        ):
            sha256(getattr(self, name), name)
        positive_int(self.input_width, "input_width")
        positive_int(self.input_height, "input_height")
        positive_int(self.batch_size, "batch_size")
        non_negative_int(self.warmup_frame_count, "warmup_frame_count")
        positive_int(self.measured_frame_count, "measured_frame_count")
        if self.inference_policy not in INFERENCE_POLICIES:
            raise ValueError(f"unsupported inference policy: {self.inference_policy}")
        if self.inference_policy == "every_frame" and self.frame_skip_interval != 1:
            raise ValueError("every_frame requires frame_skip_interval=1")
        if self.inference_policy == "every_nth_frame" and self.frame_skip_interval < 2:
            raise ValueError("every_nth_frame requires frame_skip_interval >= 2")
        if self.roi_mode != "disabled":
            raise ValueError("visual static benchmark v1 supports roi_mode=disabled")
        confidence = finite_non_negative(self.confidence_threshold, "confidence_threshold")
        if confidence > 1:
            raise ValueError("confidence_threshold must not exceed 1")
        object.__setattr__(self, "confidence_threshold", confidence)
        if self.target_inference_rate_hz is not None:
            value = finite_non_negative(self.target_inference_rate_hz, "target_inference_rate_hz")
            if value == 0:
                raise ValueError("target_inference_rate_hz must be positive")
            object.__setattr__(self, "target_inference_rate_hz", value)
        if self.random_seed is not None:
            non_negative_int(self.random_seed, "random_seed")

    def identity_record(self):
        return asdict(self)

    @property
    def condition_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "condition_identity_sha256": self.condition_identity_sha256}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("condition_identity_sha256", None)
        condition = cls(**values)
        if supplied != condition.condition_identity_sha256:
            raise ValueError("benchmark condition identity SHA256 mismatch")
        return condition

    def validate_references(self, dataset, model, preprocessing):
        if not isinstance(dataset, DatasetIdentity):
            raise TypeError("dataset must be a DatasetIdentity")
        if not isinstance(model, ModelIdentity):
            raise TypeError("model must be a ModelIdentity")
        if not isinstance(preprocessing, PreprocessingIdentity):
            raise TypeError("preprocessing must be a PreprocessingIdentity")
        checks = {
            "dataset_identity_sha256": (self.dataset_identity_sha256, dataset.dataset_identity_sha256),
            "decoder_configuration_id": (self.decoder_configuration_id, dataset.decoder_configuration_id),
            "model_identity_sha256": (self.model_identity_sha256, model.model_identity_sha256),
            "preprocessing_configuration_id": (self.preprocessing_configuration_id, preprocessing.preprocessing_configuration_id),
            "model_preprocessing_configuration_id": (model.preprocessing_configuration_id, preprocessing.preprocessing_configuration_id),
            "input_width": (self.input_width, model.input_width, preprocessing.target_input_width),
            "input_height": (self.input_height, model.input_height, preprocessing.target_input_height),
            "runtime_backend": (self.runtime_backend, model.runtime_backend),
            "precision": (self.precision, model.precision),
            "class_order_identity": (dataset.class_order_identity, model.class_order_identity),
        }
        mismatches = [name for name, values in checks.items() if len(set(values)) != 1]
        if mismatches:
            raise ValueError("visual benchmark identity mismatch: " + ", ".join(mismatches))
        return True
