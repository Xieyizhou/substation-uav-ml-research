"""Acceptance policy for a versioned labelled visual pilot."""

from __future__ import annotations

from collections import Counter


VALID_SYNC_STATUSES = frozenset({"exact", "nearest_within_tolerance"})


def validate_v3_acceptance_policy(protocol):
    recording = protocol.get("recording") or {}
    fraction = recording.get("maximum_invalid_truth_rgb_fraction")
    if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError("invalid-truth RGB fraction limit must be between 0 and 1")
    for name in (
        "maximum_consecutive_invalid_truth_rgb_frames",
        "maximum_unmatched_rgb_frames",
        "maximum_ambiguous_rgb_frames",
    ):
        value = recording.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        recording.get("invalid_truth_handling")
        != "exclude_from_dataset_and_retain_failure_record"
    ):
        raise ValueError("v3 invalid truth handling policy is not explicit")
    coverage = protocol.get("coverage") or {}
    phases = coverage.get("required_mission_phases") or []
    durations = coverage.get("mission_phase_minimum_duration_s") or {}
    if set(phases) != set(durations) or any(
        not isinstance(value, (int, float)) or value <= 0
        for value in durations.values()
    ):
        raise ValueError("v3 mission phase duration policy is invalid")


def maximum_consecutive_status(statuses, selected):
    maximum = current = 0
    for status in statuses:
        if status == selected:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def mission_phase_durations(annotations, frames_by_id, required_phases):
    timestamps = {phase: [] for phase in required_phases}
    for annotation in annotations:
        if annotation.mission_phase in timestamps:
            timestamps[annotation.mission_phase].append(
                frames_by_id[annotation.frame_id].capture_timestamp
            )
    return {
        phase: (
            max(values) - min(values)
            if len(values) >= 2
            else 0.0
        )
        for phase, values in timestamps.items()
    }


def invalid_truth_acceptance_failures(statuses, frame_count, recording):
    statuses = tuple(statuses)
    invalid_count = statuses.count("invalid_truth")
    invalid_fraction = invalid_count / frame_count if frame_count else 0.0
    maximum_fraction = float(
        recording["maximum_invalid_truth_rgb_fraction"]
    )
    failures = []
    if invalid_fraction > maximum_fraction:
        failures.append(
            "invalid-truth RGB fraction "
            f"{invalid_fraction:.6f} exceeds {maximum_fraction:.6f}"
        )
    maximum_run = int(
        recording["maximum_consecutive_invalid_truth_rgb_frames"]
    )
    observed_run = maximum_consecutive_status(statuses, "invalid_truth")
    if observed_run > maximum_run:
        failures.append(
            f"consecutive invalid-truth RGB frames {observed_run} exceed {maximum_run}"
        )
    return failures


def pilot_acceptance_failures(
    *,
    frames,
    synchronization,
    annotations,
    frames_by_id,
    metadata,
    summary,
    protocol,
    require_labelled_target=True,
):
    failures = []
    statuses = [item.synchronization_status for item in synchronization]
    counts = Counter(statuses)
    if not frames:
        failures.append("pilot contains no valid RGB frames")
    if metadata.get("invalid_frames"):
        failures.append("pilot contains invalid RGB messages")

    recording = protocol["recording"]
    for status, setting in (
        ("unmatched", "maximum_unmatched_rgb_frames"),
        ("ambiguous", "maximum_ambiguous_rgb_frames"),
    ):
        maximum = int(recording[setting])
        if counts[status] > maximum:
            failures.append(
                f"{status} RGB frames {counts[status]} exceed {maximum}"
            )

    failures.extend(
        invalid_truth_acceptance_failures(statuses, len(frames), recording)
    )

    valid_frame_ids = {
        item.rgb_frame_id
        for item in synchronization
        if item.synchronization_status in VALID_SYNC_STATUSES
    }
    annotation_ids = {annotation.frame_id for annotation in annotations}
    if annotation_ids != valid_frame_ids:
        failures.append("valid synchronized frame membership is incomplete")

    coverage = protocol["coverage"]
    phase_durations = mission_phase_durations(
        annotations,
        frames_by_id,
        coverage["required_mission_phases"],
    )
    for phase, minimum in coverage[
        "mission_phase_minimum_duration_s"
    ].items():
        duration = phase_durations.get(phase, 0.0)
        if duration + 1e-9 < float(minimum):
            failures.append(
                f"pilot mission phase duration is too short: "
                f"{phase}={duration:.3f}s, required={float(minimum):.3f}s"
            )

    sync_summary = summary.get("synchronization") or {}
    if (
        require_labelled_target
        and not sync_summary.get("labelled_target_frame_count")
    ):
        failures.append("pilot has no labelled target frame")
    valid_count = len(annotations)
    no_target_fraction = (
        sync_summary.get("no_target_frame_count", 0) / valid_count
        if valid_count
        else 0.0
    )
    minimum_no_target_fraction = float(
        coverage["minimum_no_target_fraction"]
    )
    if (
        minimum_no_target_fraction > 0.0
        and not sync_summary.get("no_target_frame_count")
    ):
        failures.append("pilot has no verified no-target frame")
    if no_target_fraction < minimum_no_target_fraction:
        failures.append("pilot no-target fraction is too small")
    return failures, phase_durations
