"""Model identity contract."""

from dataclasses import asdict, dataclass

from src.ml.artifacts import object_sha256
from src.vision.contracts.identity_validation import (
    PRECISIONS,
    VISUAL_IDENTITY_SCHEMA_VERSION,
    optional_text,
    positive_int,
    required_text,
    sha256,
)


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
            raise ValueError(f"unsupported model identity schema {self.identity_schema_version}")
        for name in ("model_family", "architecture_variant", "task", "runtime_backend", "runtime_version_identity"):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        positive_int(self.input_width, "input_width")
        positive_int(self.input_height, "input_height")
        positive_int(self.model_package_schema_version, "model_package_schema_version")
        if self.input_pixel_format != "rgb8":
            raise ValueError("visual model input_pixel_format must be rgb8")
        for name in ("class_order_identity", "preprocessing_configuration_id", "weights_sha256", "model_file_sha256"):
            sha256(getattr(self, name), name)
        for name in ("training_dataset_identity", "export_configuration_identity"):
            object.__setattr__(self, name, sha256(getattr(self, name), name, optional=True))
        object.__setattr__(self, "training_code_commit_sha", optional_text(self.training_code_commit_sha, "training_code_commit_sha"))
        if self.precision not in PRECISIONS and not self.precision.startswith("other:"):
            raise ValueError("precision must be fp32, fp16, int8, or other:<name>")
        if self.precision.startswith("other:") and not self.precision[6:].strip():
            raise ValueError("other precision must be explicitly named")
        if self.onnx_opset is not None:
            positive_int(self.onnx_opset, "onnx_opset")

    def identity_record(self):
        return asdict(self)

    @property
    def model_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "model_identity_sha256": self.model_identity_sha256}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("model_identity_sha256", None)
        identity = cls(**values)
        if supplied != identity.model_identity_sha256:
            raise ValueError("model identity SHA256 mismatch")
        return identity
