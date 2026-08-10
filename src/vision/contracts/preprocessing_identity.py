"""Preprocessing identity contract."""

from dataclasses import asdict, dataclass

from src.ml.artifacts import object_sha256
from src.vision.contracts.identity_validation import (
    VISUAL_IDENTITY_SCHEMA_VERSION,
    positive_int,
    required_text,
)


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
            raise ValueError(f"unsupported preprocessing identity schema {self.identity_schema_version}")
        positive_int(self.target_input_width, "target_input_width")
        positive_int(self.target_input_height, "target_input_height")
        for name in (
            "resize_policy", "letterbox_policy", "interpolation_method",
            "rgb_bgr_policy", "batch_dimension_policy", "tensor_dtype",
            "contiguous_memory_policy", "implementation", "implementation_version",
        ):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        padding = tuple(self.padding_value)
        if len(padding) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 255
            for value in padding
        ):
            raise ValueError("padding_value must contain three uint8 values")
        object.__setattr__(self, "padding_value", padding)
        if not isinstance(self.hwc_to_chw, bool) or not isinstance(self.uint8_to_float, bool):
            raise ValueError("preprocessing conversion policies must be boolean")
        if self.normalization_scale <= 0:
            raise ValueError("normalization_scale must be positive")
        object.__setattr__(self, "normalization_scale", float(self.normalization_scale))
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
        return {**self.identity_record(), "preprocessing_configuration_id": self.preprocessing_configuration_id}

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
