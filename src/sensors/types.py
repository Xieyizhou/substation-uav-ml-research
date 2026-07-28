"""Stable, simulator-independent sensor and perception data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any


def _finite(value: float) -> bool:
    return math.isfinite(value)


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

    def valid_ranges(self) -> tuple[float, ...]:
        return tuple(
            value
            for value in self.ranges_m
            if _finite(value) and self.range_min_m <= value <= self.range_max_m
        )

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
class PointCloudFrame:
    timestamp_s: float
    received_monotonic_s: float
    frame_id: str
    points_xyz_m: tuple[tuple[float, float, float], ...]
    source: str
    sequence: int = 0


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
    position_ned_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class PerceptionSnapshot:
    timestamp_s: float
    frame_id: str
    sensor_health: SensorHealth
    lidar_scan: LaserScanFrame | None = None
    point_cloud: PointCloudFrame | None = None
    costmap: LocalCostmap | None = None
    risk: RiskEstimate | None = None
    equipment: tuple[EquipmentDetection, ...] = field(default_factory=tuple)
    pose_ned_m: tuple[float, float, float] | None = None
    yaw_deg: float | None = None
