"""Versioned JSONL schema for LiDAR traversability research."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES, RISK_LABELS


DATASET_SCHEMA_VERSION = 1
VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ResearchSample:
    scenario_id: str
    split: str
    map_id: str
    seed: int
    timestamp_s: float
    ranges_m: tuple[float, ...]
    range_max_m: float
    velocity_ned_m_s: tuple[float, float, float]
    pose_ned_m: tuple[float, float, float]
    yaw_deg: float
    risk_label: str
    traversability: tuple[float, ...]
    equipment_labels: tuple[str, ...] = ()
    goal_ned_m: tuple[float, float, float] | None = None
    future_trajectory_ned_m: tuple[tuple[float, float, float], ...] = ()
    ground_truth_occupancy: tuple[float, ...] = ()
    collision_time_s: float | None = None
    safety_label: str | None = None
    schema_version: int = DATASET_SCHEMA_VERSION

    def validate(self):
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset schema {self.schema_version}")
        if not self.scenario_id:
            raise ValueError("scenario_id is required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {self.split}")
        if self.risk_label not in RISK_LABELS:
            raise ValueError(f"invalid risk label: {self.risk_label}")
        if self.safety_label is not None and self.safety_label not in RISK_LABELS:
            raise ValueError(f"invalid safety label: {self.safety_label}")
        if not self.ranges_m or not self.traversability:
            raise ValueError("ranges_m and traversability must not be empty")
        if self.range_max_m <= 0:
            raise ValueError("range_max_m must be positive")
        if any(label not in EQUIPMENT_CLASSES for label in self.equipment_labels):
            raise ValueError("equipment_labels contains an unsupported class")
        if any(value < 0 or value > 1 for value in self.traversability):
            raise ValueError("traversability values must be in [0, 1]")
        if any(value < 0 or value > 1 for value in self.ground_truth_occupancy):
            raise ValueError("ground-truth occupancy values must be in [0, 1]")
        if self.collision_time_s is not None and self.collision_time_s < 0:
            raise ValueError("collision_time_s must be non-negative")

    def to_record(self):
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "split": self.split,
            "map_id": self.map_id,
            "seed": self.seed,
            "timestamp_s": self.timestamp_s,
            "ranges_m": list(self.ranges_m),
            "range_max_m": self.range_max_m,
            "velocity_ned_m_s": list(self.velocity_ned_m_s),
            "pose_ned_m": list(self.pose_ned_m),
            "yaw_deg": self.yaw_deg,
            "risk_label": self.risk_label,
            "traversability": list(self.traversability),
            "equipment_labels": list(self.equipment_labels),
            "goal_ned_m": list(self.goal_ned_m) if self.goal_ned_m else None,
            "future_trajectory_ned_m": [
                list(point) for point in self.future_trajectory_ned_m
            ],
            "ground_truth_occupancy": list(self.ground_truth_occupancy),
            "collision_time_s": self.collision_time_s,
            "safety_label": self.safety_label or self.risk_label,
        }

    @classmethod
    def from_record(cls, record):
        sample = cls(
            scenario_id=str(record["scenario_id"]),
            split=str(record["split"]),
            map_id=str(record["map_id"]),
            seed=int(record["seed"]),
            timestamp_s=float(record["timestamp_s"]),
            ranges_m=tuple(float(value) for value in record["ranges_m"]),
            range_max_m=float(record["range_max_m"]),
            velocity_ned_m_s=tuple(float(value) for value in record["velocity_ned_m_s"]),
            pose_ned_m=tuple(float(value) for value in record["pose_ned_m"]),
            yaw_deg=float(record["yaw_deg"]),
            risk_label=str(record["risk_label"]),
            traversability=tuple(float(value) for value in record["traversability"]),
            equipment_labels=tuple(record.get("equipment_labels", [])),
            goal_ned_m=(
                tuple(float(value) for value in record["goal_ned_m"])
                if record.get("goal_ned_m") is not None
                else None
            ),
            future_trajectory_ned_m=tuple(
                tuple(float(value) for value in point)
                for point in record.get("future_trajectory_ned_m", [])
            ),
            ground_truth_occupancy=tuple(
                float(value) for value in record.get("ground_truth_occupancy", [])
            ),
            collision_time_s=(
                float(record["collision_time_s"])
                if record.get("collision_time_s") is not None
                else None
            ),
            safety_label=str(record.get("safety_label", record["risk_label"])),
            schema_version=int(record.get("schema_version", 0)),
        )
        sample.validate()
        return sample


def load_dataset(path: Path) -> list[ResearchSample]:
    samples = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                samples.append(ResearchSample.from_record(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not samples:
        raise ValueError(f"{path} contains no research samples")
    validate_split_isolation(samples)
    return samples


def validate_split_isolation(samples):
    scenarios_by_split = {split: set() for split in VALID_SPLITS}
    map_seed_splits = {}
    for sample in samples:
        sample.validate()
        scenarios_by_split[sample.split].add(sample.scenario_id)
        key = (sample.map_id, sample.seed)
        map_seed_splits.setdefault(key, set()).add(sample.split)
    overlaps = []
    splits = sorted(VALID_SPLITS)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            shared = scenarios_by_split[left] & scenarios_by_split[right]
            if shared:
                overlaps.append(f"{left}/{right}: {sorted(shared)[:3]}")
    leaking_seeds = [key for key, values in map_seed_splits.items() if len(values) > 1]
    if overlaps or leaking_seeds:
        details = "; ".join(overlaps)
        if leaking_seeds:
            details += f"; map/seed leakage: {leaking_seeds[:3]}"
        raise ValueError("dataset split leakage detected: " + details.strip("; "))


def normalized_scan(sample: ResearchSample, size=360):
    """Resample and normalize a scan without requiring NumPy."""
    source = [
        min(max(value, 0.0), sample.range_max_m) / sample.range_max_m
        if math.isfinite(value)
        else 1.0
        for value in sample.ranges_m
    ]
    if len(source) == size:
        return source
    return [
        source[min(round(index * (len(source) - 1) / max(size - 1, 1)), len(source) - 1)]
        for index in range(size)
    ]
