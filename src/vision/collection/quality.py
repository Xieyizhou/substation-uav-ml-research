"""Per-recording coverage gates for visual collection protocol v2."""

from collections import Counter, defaultdict


SIZE_BINS = ("small", "medium", "large")


def _size_bin(relative_area):
    if relative_area < 0.02:
        return "small"
    if relative_area < 0.10:
        return "medium"
    return "large"


def _covered_duration(timestamps, maximum_gap_s=0.1):
    ordered = sorted(set(timestamps))
    return sum(
        current - previous
        for previous, current in zip(ordered, ordered[1:])
        if current - previous <= maximum_gap_s
    )


def summarize_target_route(annotations, frames_by_id, target_class):
    size_counts = Counter()
    phase_timestamps = defaultdict(list)
    target_frame_count = 0
    truncated_frame_count = 0
    for annotation in annotations:
        targets = [item for item in annotation.objects if item.class_name == target_class]
        if not targets:
            continue
        target_frame_count += 1
        frame = frames_by_id[annotation.frame_id]
        phase_timestamps[annotation.mission_phase].append(frame.capture_timestamp)
        if any(item.truncation_status == "truncated" for item in targets):
            truncated_frame_count += 1
        relative_area = max(
            (item.bbox_xyxy[2] - item.bbox_xyxy[0])
            * (item.bbox_xyxy[3] - item.bbox_xyxy[1])
            / (annotation.image_width * annotation.image_height)
            for item in targets
        )
        size_counts[_size_bin(relative_area)] += 1
    return {
        "target_class": target_class,
        "target_frame_count": target_frame_count,
        "size_bin_frame_counts": {name: size_counts[name] for name in SIZE_BINS},
        "visible_phase_duration_s": {
            phase: _covered_duration(values)
            for phase, values in sorted(phase_timestamps.items())
        },
        "truncated_frame_count": truncated_frame_count,
        "truncated_fraction": (
            truncated_frame_count / target_frame_count if target_frame_count else None
        ),
    }


def summarize_background_route(annotations):
    labelled = sum(annotation.annotation_status == "labelled" for annotation in annotations)
    no_target = sum(annotation.annotation_status == "verified_no_target" for annotation in annotations)
    return {
        "target_class": None,
        "verified_no_target_frame_count": no_target,
        "unexpected_labelled_frame_count": labelled,
    }


def route_quality_failures(annotations, frames_by_id, row, protocol):
    gate = protocol["route_gate"]
    target_class = row.get("target_class")
    if target_class is None:
        summary = summarize_background_route(annotations)
        failures = []
        if summary["unexpected_labelled_frame_count"]:
            failures.append("background route contains labelled target frames")
        if summary["verified_no_target_frame_count"] < int(gate["minimum_target_frames"]):
            failures.append("background route has insufficient verified-no-target frames")
        return summary, failures
    summary = summarize_target_route(annotations, frames_by_id, target_class)
    failures = []
    if summary["target_frame_count"] < int(gate["minimum_target_frames"]):
        failures.append(f"{target_class} route has insufficient target frames")
    minimum_size = int(gate["minimum_frames_per_size_bin"])
    for name, count in summary["size_bin_frame_counts"].items():
        if count < minimum_size:
            failures.append(f"{target_class} route has insufficient {name} frames")
    minimum_phase = float(protocol["recording"]["minimum_phase_duration_s"])
    for phase in protocol["recording"]["required_mission_phases"]:
        if summary["visible_phase_duration_s"].get(phase, 0.0) < minimum_phase:
            failures.append(f"{target_class} is not visible long enough in {phase}")
    if (summary["truncated_fraction"] or 0.0) > float(gate["maximum_truncated_fraction"]):
        failures.append(f"{target_class} route exceeds truncated-frame limit")
    return summary, failures
