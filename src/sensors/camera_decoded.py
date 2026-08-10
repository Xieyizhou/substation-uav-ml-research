"""Canonical decoded-camera contracts, identity, hashing, and errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re

import numpy as np
from PIL import __version__ as PILLOW_VERSION

from src.sensors.types import CAMERA_RAW_LAYOUT, CameraFrame, VisualTiming


DECODED_IMAGE_SCHEMA_VERSION = 1
DECODER_CONFIGURATION_SCHEMA_VERSION = 1
DECODER_ID = "canonical-camera-decoder"
DECODER_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_OPTIONS = {
    "cpu_only": "true",
    "exif_orientation": "ignored",
    "icc_profile": "ignored",
    "palette_mode": "unsupported",
}


class CameraDecoderError(ValueError):
    """Base error carrying portable frame identity."""

    def __init__(self, frame, detail):
        if isinstance(frame, CameraFrame):
            identity = (
                f"frame_id={frame.frame_id!r} sequence={frame.sequence_number}"
            )
        else:
            frame_id, sequence_number = frame
            identity = f"frame_id={frame_id!r} sequence={sequence_number}"
        super().__init__(f"{identity}: {detail}")


class DecoderConfigurationError(ValueError):
    pass


class DecoderPayloadMissingError(CameraDecoderError):
    pass


class DecoderPayloadIntegrityError(CameraDecoderError):
    pass


class DecoderPayloadFormatMismatchError(CameraDecoderError):
    pass


class DecoderMalformedPayloadError(CameraDecoderError):
    pass


class DecoderRawByteCountError(CameraDecoderError):
    pass


class DecoderUnsupportedPixelFormatError(CameraDecoderError):
    pass


class DecoderUnsupportedDtypeError(CameraDecoderError):
    pass


class DecoderUnsupportedStrideError(CameraDecoderError):
    pass


class DecoderDimensionMismatchError(CameraDecoderError):
    pass


class DecoderContentHashError(CameraDecoderError):
    pass


class DecoderDeterminismError(CameraDecoderError):
    pass


@dataclass(frozen=True)
class DecoderConfiguration:
    schema_version: int = DECODER_CONFIGURATION_SCHEMA_VERSION
    decoder_id: str = DECODER_ID
    decoder_version: str = DECODER_VERSION
    imaging_backend: str = "Pillow"
    imaging_backend_version: str = PILLOW_VERSION
    target_pixel_format: str = "rgb8"
    target_dtype: str = "uint8"
    target_layout: str = "HWC"
    grayscale_policy: str = "repeat_luma_to_rgb"
    alpha_policy: str = "drop_alpha_without_compositing"
    raw_layout_policy: str = CAMERA_RAW_LAYOUT
    color_conversion_policy: str = "explicit_declared_layout_to_rgb"
    options: dict[str, str] = field(default_factory=lambda: dict(_SUPPORTED_OPTIONS))

    def __post_init__(self):
        expected = {
            "schema_version": DECODER_CONFIGURATION_SCHEMA_VERSION,
            "decoder_id": DECODER_ID,
            "decoder_version": DECODER_VERSION,
            "imaging_backend": "Pillow",
            "imaging_backend_version": PILLOW_VERSION,
            "target_pixel_format": "rgb8",
            "target_dtype": "uint8",
            "target_layout": "HWC",
            "grayscale_policy": "repeat_luma_to_rgb",
            "alpha_policy": "drop_alpha_without_compositing",
            "raw_layout_policy": CAMERA_RAW_LAYOUT,
            "color_conversion_policy": "explicit_declared_layout_to_rgb",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise DecoderConfigurationError(
                    f"unsupported decoder configuration {name}={getattr(self, name)!r}"
                )
        if self.options != _SUPPORTED_OPTIONS:
            raise DecoderConfigurationError(
                f"unsupported decoder options {self.options!r}"
            )

    def to_record(self):
        return asdict(self)

    @property
    def configuration_id(self):
        encoded = json.dumps(
            self.to_record(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def decoded_content_sha256(
    image_array,
    *,
    pixel_format="rgb8",
    dtype="uint8",
    layout="HWC",
):
    """Hash canonical metadata plus contiguous pixel bytes."""
    array = np.asarray(image_array)
    if array.ndim != 3:
        raise ValueError("decoded image hash requires a three-dimensional array")
    header = {
        "schema_version": DECODED_IMAGE_SCHEMA_VERSION,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "pixel_format": pixel_format,
        "dtype": dtype,
        "layout": layout,
    }
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header_bytes)
    digest.update(b"\n")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DecodedImage:
    frame_id: str
    source_id: str
    sequence_number: int
    width: int
    height: int
    pixel_format: str
    dtype: str
    layout: str
    source_payload_sha256: str
    decoded_content_sha256: str
    decoder_id: str
    decoder_version: str
    decoder_configuration_id: str
    array: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self):
        identity = (self.frame_id, self.sequence_number)
        if self.pixel_format != "rgb8":
            raise DecoderUnsupportedPixelFormatError(
                identity,
                f"canonical decoded pixel format must be rgb8, got {self.pixel_format!r}",
            )
        if self.dtype != "uint8":
            raise DecoderUnsupportedDtypeError(
                identity,
                f"canonical decoded dtype must be uint8, got {self.dtype!r}",
            )
        if self.layout != "HWC":
            raise CameraDecoderError(
                identity,
                f"canonical decoded layout must be HWC, got {self.layout!r}",
            )
        array = np.asarray(self.array)
        if array.dtype != np.uint8:
            raise DecoderUnsupportedDtypeError(
                identity, f"decoded array dtype must be uint8, got {array.dtype}"
            )
        if array.shape != (self.height, self.width, 3):
            raise DecoderDimensionMismatchError(
                identity,
                f"decoded array shape {array.shape} does not match "
                f"{self.height}x{self.width} RGB8 HWC",
            )
        canonical = np.array(array, dtype=np.uint8, order="C", copy=True)
        canonical.setflags(write=False)
        object.__setattr__(self, "array", canonical)
        for name in ("source_payload_sha256", "decoded_content_sha256"):
            if not _SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise CameraDecoderError(
                    identity, f"{name} must be a lowercase SHA256 digest"
                )
        if not _SHA256_PATTERN.fullmatch(self.decoder_configuration_id):
            raise DecoderConfigurationError(
                "decoder_configuration_id must be a lowercase SHA256 digest"
            )
        if not self.decoder_id or not self.decoder_version:
            raise DecoderConfigurationError(
                "decoder_id and decoder_version must be non-empty"
            )
        if decoded_content_sha256(canonical) != self.decoded_content_sha256:
            raise DecoderContentHashError(
                identity, "decoded-content SHA256 validation failed"
            )

    def to_metadata_record(self):
        return {
            "decoded_image_schema_version": DECODED_IMAGE_SCHEMA_VERSION,
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "sequence_number": self.sequence_number,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "dtype": self.dtype,
            "layout": self.layout,
            "source_payload_sha256": self.source_payload_sha256,
            "decoded_content_sha256": self.decoded_content_sha256,
            "decoder_id": self.decoder_id,
            "decoder_version": self.decoder_version,
            "decoder_configuration_id": self.decoder_configuration_id,
        }


@dataclass(frozen=True)
class DecodedCameraResult:
    image: DecodedImage
    timing: VisualTiming
    validation_status: str = "passed"
    warnings: tuple[str, ...] = ()
