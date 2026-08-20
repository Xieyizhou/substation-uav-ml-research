"""Deterministic temporal confirmation for ordered visual detections."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from src.ml.artifacts import file_sha256, write_json
from src.vision.evaluation.detection_metrics import box_iou, match_detections
from src.vision.evaluation.yolo_evaluation import _truth


@dataclass(frozen=True)
class TemporalObservationConfig:
    association_iou: float = 0.30
    confidence_alpha: float = 0.60
    confirmation_hits: int = 3
    confirmation_window: int = 5
    hold_ms: float = 200.0
    expiration_ms: float = 500.0

    def __post_init__(self):
        if not 0 < self.association_iou <= 1:
            raise ValueError("association_iou must be in (0, 1]")
        if not 0 < self.confidence_alpha <= 1:
            raise ValueError("confidence_alpha must be in (0, 1]")
        if not 0 < self.confirmation_hits <= self.confirmation_window:
            raise ValueError("confirmation hits must fit inside the window")
        if not 0 <= self.hold_ms <= self.expiration_ms:
            raise ValueError("hold_ms must not exceed expiration_ms")


@dataclass
class _Track:
    track_id: int
    class_name: str
    bbox: list[float]
    smoothed_confidence: float
    last_seen_s: float
    hit_frames: list[int] = field(default_factory=list)
    confirmed: bool = False


class TemporalObservationFilter:
    """Associate, smooth, confirm, hold, and expire class-aware tracks."""

    def __init__(self, config=None):
        self.config = config or TemporalObservationConfig()
        self._tracks: list[_Track] = []
        self._next_track_id = 1
        self._inference_index = -1

    def update(self, timestamp_s, predictions, *, inferred=True):
        timestamp_s = float(timestamp_s)
        if inferred:
            self._inference_index += 1
            self._associate(timestamp_s, predictions)
        self._expire(timestamp_s)
        return self._visible(timestamp_s)

    def _associate(self, timestamp_s, predictions):
        unmatched = set(range(len(predictions)))
        for track in sorted(self._tracks, key=lambda item: item.track_id):
            choices = [
                (box_iou(track.bbox, row["bbox"]), index)
                for index, row in enumerate(predictions)
                if index in unmatched and row["class_name"] == track.class_name
            ]
            overlap, index = max(choices, default=(0.0, -1))
            if overlap < self.config.association_iou:
                continue
            self._update_track(track, timestamp_s, predictions[index])
            unmatched.remove(index)
        for index in sorted(unmatched):
            self._create_track(timestamp_s, predictions[index])

    def _update_track(self, track, timestamp_s, prediction):
        alpha = self.config.confidence_alpha
        track.bbox = [float(value) for value in prediction["bbox"]]
        track.smoothed_confidence = (
            alpha * float(prediction["confidence"])
            + (1 - alpha) * track.smoothed_confidence
        )
        track.last_seen_s = timestamp_s
        track.hit_frames.append(self._inference_index)
        cutoff = self._inference_index - self.config.confirmation_window + 1
        track.hit_frames = [value for value in track.hit_frames if value >= cutoff]
        track.confirmed = track.confirmed or (
            len(track.hit_frames) >= self.config.confirmation_hits
        )

    def _create_track(self, timestamp_s, prediction):
        track = _Track(
            track_id=self._next_track_id,
            class_name=str(prediction["class_name"]),
            bbox=[float(value) for value in prediction["bbox"]],
            smoothed_confidence=float(prediction["confidence"]),
            last_seen_s=timestamp_s,
            hit_frames=[self._inference_index],
        )
        self._next_track_id += 1
        self._tracks.append(track)

    def _expire(self, timestamp_s):
        expiration_s = self.config.expiration_ms / 1000.0
        self._tracks = [
            track
            for track in self._tracks
            if timestamp_s - track.last_seen_s <= expiration_s
        ]

    def _visible(self, timestamp_s):
        hold_s = self.config.hold_ms / 1000.0
        return [
            {
                "tracking_id": f"temporal-{track.track_id}",
                "class_name": track.class_name,
                "confidence": track.smoothed_confidence,
                "bbox": list(track.bbox),
                "held": timestamp_s > track.last_seen_s,
            }
            for track in self._tracks
            if track.confirmed and timestamp_s - track.last_seen_s <= hold_s
        ]


def evaluate_temporal_observations(frames, *, config=None, threshold=0.05):
    config = config or TemporalObservationConfig()
    ordered = sorted(
        frames,
        key=lambda row: (
            str(row.get("stream_id", "default")),
            row["timestamp_s"],
            row["sample_id"],
        ),
    )
    if len({row["sample_id"] for row in ordered}) != len(ordered):
        raise ValueError("temporal input contains duplicate sample IDs")
    temporal_filter = TemporalObservationFilter(config)
    active_stream = None
    inference_truth = inference_matches = 0
    timeline_truth = timeline_matches = 0
    output = []
    for row in ordered:
        stream_id = str(row.get("stream_id", "default"))
        if active_stream != stream_id:
            temporal_filter = TemporalObservationFilter(config)
            active_stream = stream_id
        inferred = bool(row.get("inferred", True))
        temporal = temporal_filter.update(
            row["timestamp_s"], row.get("predictions", []), inferred=inferred
        )
        truth = row.get("truth", [])
        timeline_truth += len(truth)
        timeline_matches += len(
            match_detections(truth, temporal, threshold=threshold)[0]
        )
        if inferred:
            inference_truth += len(truth)
            inference_matches += len(
                match_detections(
                    truth, row.get("predictions", []), threshold=threshold
                )[0]
            )
        output.append({
            "sample_id": row["sample_id"],
            "stream_id": stream_id,
            "timestamp_s": float(row["timestamp_s"]),
            "inferred": inferred,
            "observations": temporal,
        })
    return {
        "schema_version": 1,
        "config": asdict(config),
        "threshold": float(threshold),
        "frame_count": len(ordered),
        "inference_frame_count": sum(row["inferred"] for row in output),
        "inference_frame_recall": (
            inference_matches / inference_truth if inference_truth else None
        ),
        "timeline_coverage_recall": (
            timeline_matches / timeline_truth if timeline_truth else None
        ),
        "observation_frame_count": sum(bool(row["observations"]) for row in output),
        "frames": output,
    }


def evaluate_temporal_jsonl(input_path, output_path, *, threshold=0.05):
    input_path, output_path = Path(input_path), Path(output_path)
    frames = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    result = evaluate_temporal_observations(frames, threshold=threshold)
    summary = {key: value for key, value in result.items() if key != "frames"}
    summary["input_sha256"] = file_sha256(input_path)
    frame_path = output_path.with_suffix(".frames.jsonl")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in result["frames"]
    )
    frame_path.write_text(payload, encoding="utf-8")
    summary["observations_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    write_json(output_path, summary)
    return summary


def evaluate_temporal_replay(
    predictions_path,
    membership_path,
    dataset_root,
    output_path,
    *,
    threshold=0.37,
    source_rate_hz=30.0,
    input_size=320,
):
    """Evaluate saved replay detections on the complete ordered timeline."""
    predictions_path = Path(predictions_path)
    membership_path = Path(membership_path)
    dataset_root = Path(dataset_root)
    prediction_rows = {
        row["sample_id"]: row.get("predictions", [])
        for row in _read_jsonl(predictions_path)
    }
    membership = _read_jsonl(membership_path)
    frames = []
    for index, row in enumerate(membership):
        sample_id = row["sample_id"]
        stream_id = str(row.get("recording_id", "default"))
        sequence = int(row.get("sequence_number", index))
        frames.append({
            "sample_id": sample_id,
            "stream_id": stream_id,
            "timestamp_s": sequence / float(source_rate_hz),
            "inferred": sample_id in prediction_rows,
            "truth": _truth(
                dataset_root / row["label_relative_path"],
                input_size=input_size,
            ),
            "predictions": prediction_rows.get(sample_id, []),
        })
    result = evaluate_temporal_observations(frames, threshold=threshold)
    summary = {key: value for key, value in result.items() if key != "frames"}
    summary.update({
        "temporal_replay_schema_version": 1,
        "predictions_sha256": file_sha256(predictions_path),
        "membership_sha256": file_sha256(membership_path),
        "source_rate_hz": float(source_rate_hz),
        "input_size": int(input_size),
    })
    output_path = Path(output_path)
    frame_path = output_path.with_suffix(".frames.jsonl")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in result["frames"]
    )
    frame_path.write_text(payload, encoding="utf-8")
    summary["observations_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    write_json(output_path, summary)
    return summary


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]
