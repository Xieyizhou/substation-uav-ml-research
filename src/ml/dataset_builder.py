"""Build synchronized, automatically labelled research datasets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.ml.dataset import DATASET_SCHEMA_VERSION, ResearchSample, load_dataset
from src.ml.research_recorder import ResearchDatasetWriter
from src.ml.scenarios import scenario_id, split_for
from src.ml.truth_labels import (
    TRUTH_GENERATOR_VERSION,
    apply_scan_faults,
    label_scan,
)
from src.sensors.replay import load_scan_records


DATASET_MANIFEST_VERSION = 1


def _read_jsonl(path):
    if path is None:
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def _nearest_telemetry(rows, timestamp_s):
    if not rows:
        return {}
    return min(rows, key=lambda row: abs(float(row["timestamp_s"]) - timestamp_s))


def collect_replay(
    replay_path,
    dataset_dir,
    *,
    map_id,
    target_id,
    seed,
    scenario_manifest,
    telemetry_path=None,
    goal_ned_m=None,
):
    """Append one replay scenario and rebuild its dataset manifest."""
    split = split_for(map_id, int(seed))
    scenario = scenario_id(map_id, target_id, seed)
    frames = load_scan_records(Path(replay_path))
    telemetry = _read_jsonl(telemetry_path)
    dataset_dir = Path(dataset_dir)
    samples_path = dataset_dir / "samples.jsonl"
    manifest_path = dataset_dir / "dataset_manifest.json"
    previous_scenarios = []
    if manifest_path.is_file():
        previous_scenarios = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("scenarios", [])
    generator = random.Random(int(seed) ^ 0xA5A5A5A5)
    config_hash = scenario_manifest.get("config_hash") or object_sha256(
        scenario_manifest
    )
    with ResearchDatasetWriter(samples_path) as writer:
        for frame in frames:
            state = _nearest_telemetry(telemetry, frame.timestamp_s)
            velocity = tuple(
                float(value)
                for value in state.get("velocity_ned_m_s", (0.0, 0.0, 0.0))
            )
            pose = tuple(
                float(value) for value in state.get("pose_ned_m", (0.0, 0.0, 0.0))
            )
            outage = generator.random() < float(
                scenario_manifest.get("sensor_outage_probability", 0.0)
            )
            ranges = (
                tuple(float("inf") for _ in frame.ranges_m)
                if outage
                else apply_scan_faults(
                    frame.ranges_m,
                    range_max_m=frame.range_max_m,
                    noise_stddev_m=float(
                        scenario_manifest.get("lidar_noise_stddev_m", 0.0)
                    ),
                    dropout_probability=float(
                        scenario_manifest.get("lidar_dropout_probability", 0.0)
                    ),
                    generator=generator,
                )
            )
            # Labels come from the uncorrupted simulator scan. The faulted scan
            # is the model input; safety health is recorded separately.
            labels = label_scan(frame.ranges_m, frame.range_max_m, velocity)
            if outage:
                labels["safety_label"] = "danger"
            yaw = float(state.get("yaw_deg", 0.0)) + generator.gauss(
                0.0, abs(float(scenario_manifest.get("attitude_jitter_deg", 0.0)))
            )
            writer.append(
                ResearchSample(
                    scenario_id=scenario,
                    split=split,
                    map_id=map_id,
                    seed=int(seed),
                    timestamp_s=frame.timestamp_s,
                    ranges_m=ranges,
                    range_max_m=frame.range_max_m,
                    velocity_ned_m_s=velocity,
                    pose_ned_m=pose,
                    yaw_deg=yaw,
                    risk_label=labels["risk_label"],
                    traversability=labels["traversability"],
                    goal_ned_m=tuple(goal_ned_m) if goal_ned_m else None,
                    future_trajectory_ned_m=tuple(
                        tuple(float(value) for value in point)
                        for point in state.get("future_trajectory_ned_m", [])
                    ),
                    ground_truth_occupancy=labels["ground_truth_occupancy"],
                    collision_time_s=labels["collision_time_s"],
                    safety_label=labels["safety_label"],
                    recommended_direction_deg=labels[
                        "recommended_direction_deg"
                    ],
                    sensor_data_age_ms=float(state.get("sensor_data_age_ms", 0.0)),
                    sensor_healthy=bool(state.get("sensor_healthy", True)) and not outage,
                    scenario_config_hash=config_hash,
                )
            )
    manifest = build_dataset_manifest(samples_path)
    current = {
        "scenario_id": scenario,
        "map_id": map_id,
        "target_id": target_id,
        "seed": int(seed),
        "config_hash": config_hash,
    }
    by_id = {item["scenario_id"]: item for item in previous_scenarios}
    by_id[scenario] = current
    manifest["scenarios"] = [by_id[key] for key in sorted(by_id)]
    write_json(manifest_path, manifest)
    return manifest


def build_dataset_manifest(samples_path, *, root=None):
    samples_path = Path(samples_path)
    samples = load_dataset(samples_path)
    for sample in samples:
        expected = split_for(sample.map_id, sample.seed)
        if sample.split != expected:
            raise ValueError(
                f"{sample.scenario_id}: split {sample.split!r} does not match "
                f"the versioned seed policy ({expected!r})"
            )
    split_counts = Counter(sample.split for sample in samples)
    label_counts = Counter(sample.risk_label for sample in samples)
    data_hash = file_sha256(samples_path)
    identity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "data_sha256": data_hash,
        "truth_generator_version": TRUTH_GENERATOR_VERSION,
    }
    return {
        "manifest_schema_version": DATASET_MANIFEST_VERSION,
        "dataset_id": f"lidar-{object_sha256(identity)[:12]}",
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "generator_commit": git_commit(root),
        "truth_generator_version": TRUTH_GENERATOR_VERSION,
        "samples_file": samples_path.name,
        "samples": len(samples),
        "data_sha256": data_hash,
        "split_counts": dict(sorted(split_counts.items())),
        "risk_label_distribution": dict(sorted(label_counts.items())),
        "maps": sorted({sample.map_id for sample in samples}),
        "seeds": sorted({sample.seed for sample in samples}),
        "scenario_config_hashes": sorted(
            {sample.scenario_config_hash for sample in samples if sample.scenario_config_hash}
        ),
        "scenarios": [],
    }


def validate_dataset_directory(dataset_dir):
    dataset_dir = Path(dataset_dir)
    samples_path = dataset_dir / "samples.jsonl"
    manifest_path = dataset_dir / "dataset_manifest.json"
    current = build_dataset_manifest(samples_path)
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest.json is missing")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("data_sha256", "dataset_id", "dataset_schema_version"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"dataset manifest mismatch: {key}")
    return saved


def summarize_dataset(dataset_dir):
    manifest = validate_dataset_directory(dataset_dir)
    return {
        key: manifest[key]
        for key in (
            "dataset_id",
            "samples",
            "split_counts",
            "risk_label_distribution",
            "maps",
            "seeds",
            "data_sha256",
        )
    }
