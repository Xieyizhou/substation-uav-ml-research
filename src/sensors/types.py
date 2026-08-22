"""Stable, simulator-independent sensor and perception data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import PurePosixPath
import re
import time
from typing import Any


CAMERA_PAYLOAD_FORMATS = frozenset({"png", "jpeg", "raw"})
CAMERA_PIXEL_FORMATS = frozenset(
    {"rgb8", "bgr8", "rgba8", "bgra8", "mono8"}
)
CAMERA_PIXEL_CHANNELS = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
}
CAMERA_RAW_LAYOUT = "tightly_packed_uint8_no_row_padding"
LOCAL_MONOTONIC_CLOCK = "local_monotonic"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _json_serializable_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only JSON-serializable values") from error


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class SensorHealth:
    source: str
    healthy: bool
    message: str = ""
    frequency_hz: float = 0.0
    dropped_frames: int = 0
    last_frame_age_s: float | None = None


@dataclass(frozen=True)
class LaserScanFrame:
    timestamp_s: float
    received_monotonic_s: float
    frame_id: str
    angle_min_rad: float
    angle_max_rad: float
    angle_step_rad: float
    range_min_m: float
    range_max_m: float
    ranges_m: tuple[float, ...]
    source: str
    sequence: int = 0

    def age_s(self, now_s: float | None = None) -> float:
        now_s = time.monotonic() if now_s is None else now_s
        return max(0.0, now_s - self.received_monotonic_s)

    def angle_at(self, index: int) -> float:
        return self.angle_min_rad + index * self.angle_step_rad

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["ranges_m"] = list(self.ranges_m)
        record["record_type"] = "laser_scan_2d"
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "LaserScanFrame":
        values = dict(record)
        values.pop("record_type", None)
        values["ranges_m"] = tuple(float(value) for value in values["ranges_m"])
        return cls(**values)


@dataclass(frozen=True)
class CameraFrame:
    """Portable metadata for one camera payload stored outside JSONL."""

    frame_id: str
    source_id: str
    sequence_number: int
    capture_timestamp: float
    capture_clock_domain: str
    receive_monotonic_timestamp: float
    width: int
    height: int
    payload_format: str
    pixel_format: str
    payload_relative_path: str
    payload_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not str(self.frame_id).strip():
            raise ValueError("frame_id must not be empty")
        if not str(self.source_id).strip():
            raise ValueError("source_id must not be empty")
        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise ValueError("sequence_number must be a non-negative integer")
        _finite_number(self.capture_timestamp, "capture_timestamp")
        _finite_number(
            self.receive_monotonic_timestamp, "receive_monotonic_timestamp"
        )
        if self.receive_monotonic_timestamp < 0:
            raise ValueError("receive_monotonic_timestamp must be non-negative")
        if not str(self.capture_clock_domain).strip():
            raise ValueError("capture_clock_domain must not be empty")
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or self.width <= 0
        ):
            raise ValueError("width must be a positive integer")
        if (
            isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.height <= 0
        ):
            raise ValueError("height must be a positive integer")
        normalized_payload_format = str(self.payload_format).strip().lower()
        if normalized_payload_format not in CAMERA_PAYLOAD_FORMATS:
            raise ValueError(
                f"unsupported camera payload_format: {self.payload_format!r}"
            )
        object.__setattr__(self, "payload_format", normalized_payload_format)
        normalized_pixel_format = str(self.pixel_format).strip().lower()
        if normalized_pixel_format not in CAMERA_PIXEL_FORMATS:
            raise ValueError(
                f"unsupported camera pixel_format: {self.pixel_format!r}"
            )
        object.__setattr__(self, "pixel_format", normalized_pixel_format)
        payload_path = str(self.payload_relative_path)
        portable_path = PurePosixPath(payload_path)
        if (
            not payload_path
            or "\\" in payload_path
            or portable_path.is_absolute()
            or ".." in portable_path.parts
            or portable_path.name in {"", ".", ".."}
        ):
            raise ValueError("payload_relative_path must be a portable relative path")
        if not _SHA256_PATTERN.fullmatch(str(self.payload_sha256)):
            raise ValueError("payload_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "payload_sha256", self.payload_sha256.lower())
        _json_serializable_mapping(self.metadata, "metadata")

    def to_record(self):
        return {
            "record_type": "camera_frame",
            **asdict(self),
        }

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        record_type = values.pop("record_type", "camera_frame")
        if record_type != "camera_frame":
            raise ValueError(f"unsupported camera record type: {record_type!r}")
        return cls(**values)

    def capture_to_receive_ms(self):
        """Return latency only when both timestamps use local monotonic time."""
        if self.capture_clock_domain != LOCAL_MONOTONIC_CLOCK:
            return None
        duration_ms = (
            self.receive_monotonic_timestamp - self.capture_timestamp
        ) * 1000.0
        return duration_ms if duration_ms >= 0 else None


@dataclass(frozen=True)
class DepthFrame:
    """Metadata for a tightly packed little-endian float32 depth payload."""

    frame_id: str
    source_id: str
    sequence_number: int
    capture_timestamp: float
    capture_clock_domain: str
    receive_monotonic_timestamp: float
    width: int
    height: int
    payload_relative_path: str
    payload_sha256: str
    valid_depth_count: int

    def __post_init__(self):
        if not self.frame_id or not self.source_id:
            raise ValueError("depth frame identifiers must not be empty")
        if self.sequence_number < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("invalid depth frame sequence or dimensions")
        _finite_number(self.capture_timestamp, "capture_timestamp")
        _finite_number(self.receive_monotonic_timestamp, "receive_monotonic_timestamp")
        path = PurePosixPath(str(self.payload_relative_path))
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("depth payload path must be portable and relative")
        if not _SHA256_PATTERN.fullmatch(str(self.payload_sha256)):
            raise ValueError("depth payload_sha256 must be a SHA256 digest")
        if not 0 <= self.valid_depth_count <= self.width * self.height:
            raise ValueError("invalid valid_depth_count")

    def to_record(self):
        return {"record_type": "depth_frame", **asdict(self)}


@dataclass(frozen=True)
class LocalCostmap:
    timestamp_s: float
    frame_id: str
    resolution_m: float
    width: int
    height: int
    origin_forward_m: float
    origin_left_m: float
    occupancy: tuple[float, ...]
    traversability: tuple[float, ...]
    unknown: tuple[bool, ...]
    version: int = 1

    def __post_init__(self):
        cell_count = self.width * self.height
        if cell_count <= 0:
            raise ValueError("costmap width and height must be positive")
        for name, values in (
            ("occupancy", self.occupancy),
            ("traversability", self.traversability),
            ("unknown", self.unknown),
        ):
            if len(values) != cell_count:
                raise ValueError(f"{name} length must equal width * height")

    def index(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError((x, y))
        return y * self.width + x


@dataclass(frozen=True)
class RiskEstimate:
    level: str
    confidence: float
    nearest_distance_m: float | None
    collision_time_s: float | None
    recommended_direction_deg: float | None
    inference_latency_ms: float
    model_id: str
    reason: str


@dataclass(frozen=True)
class EquipmentDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    timestamp_s: float
    frame_id: str
    tracking_id: str | None = None


@dataclass(frozen=True)
class VisualTimingContext:
    """Comparable monotonic timestamps supplied to one detector call."""

    frame: CameraFrame | None = None
    queue_entered_monotonic_timestamp: float | None = None
    queue_depth: int | None = None
    deadline_ms: float | None = None

    def __post_init__(self):
        for name in ("queue_entered_monotonic_timestamp", "deadline_ms"):
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
        if self.queue_depth is not None and (
            isinstance(self.queue_depth, bool)
            or not isinstance(self.queue_depth, int)
            or self.queue_depth < 0
        ):
            raise ValueError("queue_depth must be a non-negative integer")


@dataclass(frozen=True)
class VisualTiming:
    """Detector-neutral stage timings; unavailable measurements remain null."""

    capture_to_receive_ms: float | None = None
    queue_wait_ms: float | None = None
    payload_load_ms: float | None = None
    decode_ms: float | None = None
    preprocess_ms: float | None = None
    backend_call_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None
    decision_finalize_ms: float | None = None
    end_to_end_ms: float | None = None
    queue_depth: int | None = None
    deadline_ms: float | None = None
    deadline_missed: bool | None = None
    model_id: str | None = None
    runtime_backend: str | None = None
    device: str | None = None
    input_width: int | None = None
    input_height: int | None = None
    detection_count: int | None = None
    timing_provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        duration_fields = (
            "capture_to_receive_ms",
            "queue_wait_ms",
            "payload_load_ms",
            "decode_ms",
            "preprocess_ms",
            "backend_call_ms",
            "inference_ms",
            "postprocess_ms",
            "decision_finalize_ms",
            "end_to_end_ms",
            "deadline_ms",
        )
        for name in duration_fields:
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
        for name in ("queue_depth", "detection_count"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("input_width", "input_height"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.deadline_missed is not None and not isinstance(
            self.deadline_missed, bool
        ):
            raise ValueError("deadline_missed must be a boolean or null")
        _json_serializable_mapping(self.timing_provenance, "timing_provenance")

    def to_record(self):
        return asdict(self)


@dataclass(frozen=True)
class VisualDetectionResult:
    detections: tuple[EquipmentDetection, ...]
    timing: VisualTiming
