"""Deterministic risk overlays for bootstrapping a LiDAR sandbox candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.dataset import ResearchSample
from src.ml.dataset_builder import build_dataset_manifest
from src.ml.research_recorder import ResearchDatasetWriter
from src.ml.scenarios import scenario_id, split_for
from src.ml.truth_labels import label_scan
from src.sensors.replay import load_scan_records


OVERLAY_VERSION = "deterministic-risk-overlay-v1"
DISTANCE_BY_LABEL = {"clear": 2.5, "warning": 1.0, "danger": 0.3}


@dataclass(frozen=True)
class OverlaySource:
    path: Path
    map_id: str
    seed: int

    @classmethod
    def parse(cls, value: str):
        parts = value.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError("source must use PATH:MAP:SEED")
        return cls(Path(parts[0]), parts[1], int(parts[2]))


def _selected_frames(frames, limit):
    count = min(len(frames), limit)
    if count == len(frames):
        return frames
    return [
        frames[min((index * len(frames)) // count, len(frames) - 1)]
        for index in range(count)
    ]


def _overlay_ranges(frame, label, sample_index):
    distance = DISTANCE_BY_LABEL[label]
    ranges = [
        max(float(value), DISTANCE_BY_LABEL["clear"])
        if value == value and value != float("inf")
        else frame.range_max_m
        for value in frame.ranges_m
    ]
    if label == "clear":
        return tuple(ranges)
    width = max(3, len(ranges) // 72)
    margin = width + 1
    center = margin + ((sample_index * 37) % max(len(ranges) - 2 * margin, 1))
    for index in range(max(0, center - width), min(len(ranges), center + width + 1)):
        ranges[index] = distance
    return tuple(ranges)


def _sample(frame, source, label, index, config_hash):
    ranges = _overlay_ranges(frame, label, index)
    labels = label_scan(ranges, frame.range_max_m, (0.0, 0.0, 0.0))
    if labels["risk_label"] != label:
        raise ValueError(f"overlay produced {labels['risk_label']} instead of {label}")
    split = split_for(source.map_id, source.seed)
    return ResearchSample(
        scenario_id=scenario_id(source.map_id, "risk_overlay", source.seed),
        split=split,
        map_id=source.map_id,
        seed=source.seed,
        timestamp_s=frame.timestamp_s + (list(DISTANCE_BY_LABEL).index(label) * 1e-6),
        ranges_m=ranges,
        range_max_m=frame.range_max_m,
        velocity_ned_m_s=(0.0, 0.0, 0.0),
        pose_ned_m=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        risk_label=label,
        traversability=labels["traversability"],
        ground_truth_occupancy=labels["ground_truth_occupancy"],
        collision_time_s=labels["collision_time_s"],
        safety_label=label,
        recommended_direction_deg=labels["recommended_direction_deg"],
        scenario_config_hash=config_hash,
    )


def materialize_overlay_dataset(sources, output_dir, *, max_frames_per_source=1200):
    """Create a split-safe development dataset without modifying source recordings."""
    if max_frames_per_source <= 0:
        raise ValueError("max frames per source must be positive")
    sources = tuple(sources)
    if not sources:
        raise ValueError("at least one replay source is required")
    output_dir = Path(output_dir)
    samples_path = output_dir / "samples.jsonl"
    if samples_path.exists():
        raise ValueError("output dataset already exists")
    source_records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with ResearchDatasetWriter(samples_path) as writer:
        for source in sources:
            frames = _selected_frames(load_scan_records(source.path), max_frames_per_source)
            source_hash = file_sha256(source.path)
            config_hash = object_sha256(
                {"version": OVERLAY_VERSION, "source_sha256": source_hash}
            )
            for index, frame in enumerate(frames):
                for label in DISTANCE_BY_LABEL:
                    writer.append(_sample(frame, source, label, index, config_hash))
            source_records.append(
                {
                    "path": source.path.name,
                    "map_id": source.map_id,
                    "seed": source.seed,
                    "source_sha256": source_hash,
                    "selected_frame_count": len(frames),
                    "scenario_config_hash": config_hash,
                }
            )
    manifest = build_dataset_manifest(samples_path)
    manifest.update(
        {
            "dataset_role": "development_overlay",
            "overlay_version": OVERLAY_VERSION,
            "source_recordings": source_records,
            "scenarios": [
                {
                    "scenario_id": scenario_id(row["map_id"], "risk_overlay", row["seed"]),
                    "map_id": row["map_id"],
                    "target_id": "risk_overlay",
                    "seed": row["seed"],
                    "config_hash": row["scenario_config_hash"],
                }
                for row in source_records
            ],
        }
    )
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest
