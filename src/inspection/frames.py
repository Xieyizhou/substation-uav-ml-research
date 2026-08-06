"""Paginated recording frames and scenario progress read models."""

from __future__ import annotations

from collections import Counter
from urllib.parse import quote

from src.inspection.config import AccessDenied, InspectionConfig
from src.inspection.dashboard import is_blind, load_plan
from src.inspection.models import FramePage, FrameView, ProgressView
from src.inspection.readers import iter_jsonl, jsonl_count, jsonl_page
from src.vision.contracts.annotations import VisualFrameAnnotation


def recording_row(config, recording_id):
    row = next((row for row in load_plan(config)["scenarios"]
                if row.get("recording_id") == recording_id), None)
    if row is None:
        raise AccessDenied("recording is not present in the approved plan")
    if is_blind(row):
        raise AccessDenied("blind recording details are not exposed")
    return row


def frame_page(config: InspectionConfig, recording_id: str, page=1, page_size=24):
    recording_row(config, recording_id)
    page, page_size = int(page), int(page_size)
    if page < 1 or page_size < 1 or page_size > 100:
        raise ValueError("invalid frame page")
    root = config.recording(recording_id)
    frames_path = root / "frames.jsonl"
    offset = (page - 1) * page_size
    rows = jsonl_page(frames_path, offset, page_size)
    frame_ids = {row.get("frame_id") for row in rows}
    sync = _indexed(root / "synchronization.jsonl", "rgb_frame_id", frame_ids)
    annotations = _annotations(root / "annotations.jsonl", frame_ids)
    views = tuple(_frame_view(recording_id, row, sync, annotations) for row in rows)
    return FramePage(recording_id, page, page_size, jsonl_count(frames_path), views)


def _indexed(path, key, selected=None):
    if not path.is_file():
        return {}
    return {row.get(key): row for row in iter_jsonl(path)
            if selected is None or row.get(key) in selected}


def _annotations(path, selected=None):
    result = {}
    if not path.is_file():
        return result
    for row in iter_jsonl(path):
        if selected is not None and row.get("frame_id") not in selected:
            continue
        try:
            annotation = VisualFrameAnnotation.from_record(row)
        except (TypeError, ValueError, KeyError):
            continue
        result[annotation.frame_id] = annotation
    return result


def _frame_view(recording_id, row, sync, annotations):
    frame_id = str(row.get("frame_id", ""))
    annotation = annotations.get(frame_id)
    objects = () if annotation is None else tuple(
        item for item in annotation.objects if item.validation_status != "rejected"
    )
    boxes = tuple({"class_name": item.class_name, "bbox": item.bbox_xyxy,
                   "validation": item.validation_status} for item in objects)
    status = "unavailable" if annotation is None else annotation.annotation_status
    summary = status if not boxes else f"{status}: {len(boxes)} object(s)"
    synchronization = sync.get(frame_id, {})
    return FrameView(
        frame_id, int(row.get("sequence_number", 0)),
        float(row.get("capture_timestamp", 0.0)),
        "unknown" if annotation is None else annotation.mission_phase,
        str(synchronization.get("synchronization_status", "unavailable")),
        status, summary, int(row.get("width", 0)), int(row.get("height", 0)),
        f"/api/frame/{quote(recording_id, safe='')}/{quote(frame_id, safe='')}", boxes,
    )


def frame_path(config, recording_id, frame_id):
    recording_row(config, recording_id)
    for row in iter_jsonl(config.recording(recording_id) / "frames.jsonl"):
        if row.get("frame_id") == frame_id:
            return config.frame_payload(recording_id, str(row.get("payload_relative_path", "")))
    raise FileNotFoundError("frame is not present in the recording manifest")


def scenario_progress(config, recording_id):
    row = recording_row(config, recording_id)
    root = config.recording(recording_id)
    frames = tuple(iter_jsonl(root / "frames.jsonl")) if (root / "frames.jsonl").is_file() else ()
    annotations = _annotations(root / "annotations.jsonl")
    truth_counts = Counter("unmatched_or_ambiguous" for item in _sync_uncertain(root))
    truth_counts["verified_no_target"] = sum(
        item.annotation_status == "verified_no_target" for item in annotations.values()
    )
    events = _event_views(root / "mission_events.jsonl")
    phases = _phases(annotations, frames)
    receipt = root / "identity/collection_validation.json"
    elapsed = None if len(frames) < 2 else float(frames[-1]["capture_timestamp"]) - float(frames[0]["capture_timestamp"])
    return ProgressView(recording_id, len(frames), elapsed, row.get("target_class"),
                        "present" if receipt.is_file() else "missing", phases, events, dict(truth_counts))


def _event_views(path):
    if not path.is_file():
        return ()
    return tuple({
        "event": row.get("event_type", row.get("mission_phase", "event")),
        "phase": row.get("mission_phase", row.get("phase")),
        "simulation_timestamp": row.get("simulation_timestamp"),
        "status": row.get("status"),
    } for row in iter_jsonl(path))


def _sync_uncertain(root):
    path = root / "synchronization.jsonl"
    return () if not path.is_file() else tuple(
        row for row in iter_jsonl(path)
        if row.get("synchronization_status") in {"unmatched", "ambiguous", "invalid_truth"}
    )


def _phases(annotations, frames):
    timestamps = {row.get("frame_id"): row.get("capture_timestamp") for row in frames}
    grouped = {}
    for item in annotations.values():
        values = grouped.setdefault(item.mission_phase, [])
        if item.frame_id in timestamps:
            values.append(float(timestamps[item.frame_id]))
    return tuple({"phase": name, "start": min(values), "end": max(values), "frames": len(values)}
                 for name, values in grouped.items() if values)
