"""Deterministic Gazebo RGB-to-truth timestamp synchronization."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass

from src.vision.collection.gazebo_truth import SimulatorTruthFrame
from src.vision.contracts.annotations import VisualFrameAnnotation
from src.sensors.camera_decoded import DecodedImage
from src.sensors.types import CameraFrame


MATCHING_POLICY_VERSION = 1
DEFAULT_MAX_SKEW_MS = 33.334
SYNC_STATUSES = frozenset(
    {
        "exact",
        "nearest_within_tolerance",
        "unmatched",
        "ambiguous",
        "invalid_truth",
    }
)


@dataclass(frozen=True)
class SynchronizationRecord:
    rgb_frame_id: str
    rgb_sequence_number: int
    rgb_simulation_timestamp: float
    truth_message_identity: str | None
    truth_simulation_timestamp: float | None
    synchronization_offset_ms: float | None
    synchronization_status: str
    maximum_permitted_skew_ms: float
    matching_policy_version: int = MATCHING_POLICY_VERSION

    def __post_init__(self):
        if self.synchronization_status not in SYNC_STATUSES:
            raise ValueError("unsupported synchronization status")
        if self.maximum_permitted_skew_ms < 0:
            raise ValueError("maximum permitted skew must be non-negative")
        matched = self.synchronization_status != "unmatched"
        values = (
            self.truth_message_identity,
            self.truth_simulation_timestamp,
            self.synchronization_offset_ms,
        )
        if matched and any(value is None for value in values):
            raise ValueError("non-unmatched synchronization requires truth identity")
        if not matched and any(value is not None for value in values):
            raise ValueError("unmatched synchronization cannot reference truth")

    def to_record(self):
        return asdict(self)


@dataclass(frozen=True)
class SynchronizedFrame:
    frame: CameraFrame
    synchronization: SynchronizationRecord
    truth: SimulatorTruthFrame | None


def _candidate_key(frame, truth):
    offset = truth.simulation_timestamp - frame.capture_timestamp
    return (
        abs(offset),
        truth.simulation_timestamp,
        truth.message_id,
    )


def _truth_index(truths):
    grouped = defaultdict(list)
    for truth in truths:
        grouped[
            (truth.clock_domain, truth.image_width, truth.image_height)
        ].append(truth)
    index = {}
    for key, compatible in grouped.items():
        compatible.sort(
            key=lambda truth: (truth.simulation_timestamp, truth.message_id)
        )
        index[key] = (
            tuple(compatible),
            tuple(truth.simulation_timestamp for truth in compatible),
        )
    return index


def _nearest_truths(frame, compatible, timestamps):
    position = bisect_left(timestamps, frame.capture_timestamp)
    nearest_timestamps = []
    if position:
        nearest_timestamps.append(timestamps[position - 1])
    if position < len(timestamps):
        nearest_timestamps.append(timestamps[position])
    selected_distance = min(
        abs(timestamp - frame.capture_timestamp)
        for timestamp in nearest_timestamps
    )
    tied = []
    for timestamp in nearest_timestamps:
        if (
            abs(
                abs(timestamp - frame.capture_timestamp)
                - selected_distance
            )
            > 1e-12
        ):
            continue
        start = bisect_left(timestamps, timestamp)
        end = bisect_right(timestamps, timestamp)
        tied.extend(compatible[start:end])
    tied.sort(key=lambda truth: _candidate_key(frame, truth))
    return tied, selected_distance


def synchronize_visual_frames(
    frames,
    truths,
    *,
    maximum_skew_ms=DEFAULT_MAX_SKEW_MS,
):
    maximum_skew_ms = float(maximum_skew_ms)
    if maximum_skew_ms < 0:
        raise ValueError("maximum_skew_ms must be non-negative")
    index = _truth_index(truths)
    results = []
    for frame in frames:
        compatible_entry = index.get(
            (frame.capture_clock_domain, frame.width, frame.height)
        )
        if compatible_entry is None:
            record = SynchronizationRecord(
                rgb_frame_id=frame.frame_id,
                rgb_sequence_number=frame.sequence_number,
                rgb_simulation_timestamp=frame.capture_timestamp,
                truth_message_identity=None,
                truth_simulation_timestamp=None,
                synchronization_offset_ms=None,
                synchronization_status="unmatched",
                maximum_permitted_skew_ms=maximum_skew_ms,
            )
            results.append(SynchronizedFrame(frame, record, None))
            continue
        compatible, timestamps = compatible_entry
        tied, selected_distance = _nearest_truths(
            frame, compatible, timestamps
        )
        selected = tied[0]
        if selected_distance * 1000.0 > maximum_skew_ms:
            record = SynchronizationRecord(
                rgb_frame_id=frame.frame_id,
                rgb_sequence_number=frame.sequence_number,
                rgb_simulation_timestamp=frame.capture_timestamp,
                truth_message_identity=None,
                truth_simulation_timestamp=None,
                synchronization_offset_ms=None,
                synchronization_status="unmatched",
                maximum_permitted_skew_ms=maximum_skew_ms,
            )
            results.append(SynchronizedFrame(frame, record, None))
            continue
        offset_ms = (
            selected.simulation_timestamp - frame.capture_timestamp
        ) * 1000.0
        if len(tied) > 1:
            status = "ambiguous"
            linked_truth = None
        elif not selected.valid:
            status = "invalid_truth"
            linked_truth = None
        elif selected_distance <= 1e-12:
            status = "exact"
            linked_truth = selected
        else:
            status = "nearest_within_tolerance"
            linked_truth = selected
        record = SynchronizationRecord(
            rgb_frame_id=frame.frame_id,
            rgb_sequence_number=frame.sequence_number,
            rgb_simulation_timestamp=frame.capture_timestamp,
            truth_message_identity=selected.message_id,
            truth_simulation_timestamp=selected.simulation_timestamp,
            synchronization_offset_ms=offset_ms,
            synchronization_status=status,
            maximum_permitted_skew_ms=maximum_skew_ms,
        )
        results.append(SynchronizedFrame(frame, record, linked_truth))
    return tuple(results)


def materialize_frame_annotation(
    synchronized,
    decoded_image,
    *,
    recording_id,
    scenario_id,
    map_id,
    seed,
    mission_phase,
    frame_order_reference,
):
    if not isinstance(decoded_image, DecodedImage):
        raise TypeError("decoded_image must be a DecodedImage")
    truth = synchronized.truth
    if truth is None:
        return None
    annotation_status = "labelled" if truth.objects else "verified_no_target"
    annotation = VisualFrameAnnotation(
        frame_id=synchronized.frame.frame_id,
        source_id=synchronized.frame.source_id,
        sequence_number=synchronized.frame.sequence_number,
        payload_sha256=synchronized.frame.payload_sha256,
        decoded_content_sha256=decoded_image.decoded_content_sha256,
        image_width=decoded_image.width,
        image_height=decoded_image.height,
        scenario_id=scenario_id,
        map_id=map_id,
        seed=seed,
        mission_phase=mission_phase,
        frame_order_reference=frame_order_reference,
        annotation_status=annotation_status,
        objects=truth.objects,
        recording_id=recording_id,
    )
    annotation.validate_linkage(synchronized.frame, decoded_image)
    return annotation
