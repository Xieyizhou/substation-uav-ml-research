"""Gazebo bounding-box truth conversion with locked visual class semantics."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256
from src.vision.contracts.annotations import VisualObjectAnnotation
from src.sensors.gazebo_visual_transport import (
    GAZEBO_SIM_CLOCK,
    GAZEBO_JSON_STREAM_LIMIT_BYTES,
    TRUTH_CONFIGURED_TOPIC,
    TRUTH_MESSAGE_TYPE,
    discover_configured_topic,
    header_sequence,
    inspect_topic,
    message_timestamp,
    stop_stream_process,
    transport_environment,
)


SIMULATOR_LABELS = {
    1: "transformer",
    2: "switchgear",
    3: "capacitor_bank",
    4: "reactor",
}


@dataclass(frozen=True)
class SimulatorTruthFrame:
    message_id: str
    source_id: str
    source_sequence: int
    sequence_provenance: str
    simulation_timestamp: float
    clock_domain: str
    image_width: int
    image_height: int
    objects: tuple[VisualObjectAnnotation, ...]
    validation_status: str
    invalid_reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.message_id or not self.source_id:
            raise ValueError("truth message and source IDs must not be empty")
        if self.source_sequence < 0:
            raise ValueError("truth source sequence must be non-negative")
        if self.simulation_timestamp < 0:
            raise ValueError("truth simulation timestamp must be non-negative")
        if self.clock_domain != GAZEBO_SIM_CLOCK:
            raise ValueError("truth frame must use the Gazebo simulation clock")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("truth image dimensions must be positive")
        if self.validation_status not in {"valid", "invalid"}:
            raise ValueError("unsupported truth validation status")
        if self.validation_status == "invalid" and not self.invalid_reasons:
            raise ValueError("invalid truth requires at least one reason")
        for item in self.objects:
            item.validate_dimensions(self.image_width, self.image_height)

    @property
    def valid(self):
        return self.validation_status == "valid"

    @property
    def identity_sha256(self):
        return object_sha256(self.to_record())

    def to_record(self):
        record = asdict(self)
        record["objects"] = [item.to_record() for item in self.objects]
        record["invalid_reasons"] = list(self.invalid_reasons)
        return record


@dataclass(frozen=True)
class TruthSourceEvent:
    receive_index: int
    truth: SimulatorTruthFrame


def _number(container, *keys):
    for key in keys:
        if key in container:
            return float(container[key])
    # Protobuf JSON omits scalar fields whose value is the numeric default.
    return 0.0


def _box_corners(item):
    box = item.get("box")
    if not isinstance(box, dict):
        raise ValueError("annotated truth item is missing box")
    minimum = box.get("min_corner", box.get("minCorner"))
    maximum = box.get("max_corner", box.get("maxCorner"))
    if not isinstance(minimum, dict) or not isinstance(maximum, dict):
        raise ValueError("annotated truth box is missing corners")
    return (
        _number(minimum, "x"),
        _number(minimum, "y"),
        _number(maximum, "x"),
        _number(maximum, "y"),
    )


def _annotation(item, *, index, width, height, message_id):
    try:
        simulator_label = int(item.get("label"))
    except (TypeError, ValueError) as error:
        raise ValueError("truth item label must be an integer") from error
    if simulator_label not in SIMULATOR_LABELS:
        raise ValueError(f"unknown simulator label: {simulator_label}")
    class_name = SIMULATOR_LABELS[simulator_label]
    original = _box_corners(item)
    x_min, y_min, x_max, y_max = original
    clipped = (
        max(0.0, min(float(width), x_min)),
        max(0.0, min(float(height), y_min)),
        max(0.0, min(float(width), x_max)),
        max(0.0, min(float(height), y_max)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("truth bounding box has zero visible area after clipping")
    return VisualObjectAnnotation(
        annotation_id=f"{message_id}-box-{index:04d}",
        class_id=EQUIPMENT_CLASSES.index(class_name),
        class_name=class_name,
        bbox_xyxy=clipped,
        visibility_status="unknown",
        truncation_status=(
            "truncated" if clipped != original else "not_truncated"
        ),
        truth_source="gazebo_bounding_box_sensor",
        validation_status="validated",
    )


def parse_gazebo_truth_message(
    message,
    *,
    topic,
    width,
    height,
    receive_index,
):
    timestamp = message_timestamp(message)
    header_value = header_sequence(message)
    sequence = receive_index if header_value is None else header_value
    provenance = "local_receive_ordinal" if header_value is None else "gazebo_header"
    message_id = f"gazebo-truth-{sequence:09d}-{receive_index:09d}"
    items = message.get("annotated_box", message.get("annotatedBox", []))
    if not isinstance(items, list):
        raise ValueError("Gazebo truth annotated_box must be a list")
    objects = []
    invalid_reasons = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            invalid_reasons.append(f"box {index}: item must be an object")
            continue
        try:
            objects.append(
                _annotation(
                    item,
                    index=index,
                    width=width,
                    height=height,
                    message_id=message_id,
                )
            )
        except (TypeError, ValueError) as error:
            invalid_reasons.append(f"box {index}: {error}")
    return SimulatorTruthFrame(
        message_id=message_id,
        source_id=f"gazebo_truth:{topic}",
        source_sequence=sequence,
        sequence_provenance=provenance,
        simulation_timestamp=timestamp,
        clock_domain=GAZEBO_SIM_CLOCK,
        image_width=int(width),
        image_height=int(height),
        objects=tuple(objects),
        validation_status="invalid" if invalid_reasons else "valid",
        invalid_reasons=tuple(invalid_reasons),
    )


class GazeboTruthSource:
    source_id = "gazebo_bounding_box_truth"

    def __init__(self, *, width, height, topic="auto"):
        self.width = int(width)
        self.height = int(height)
        self.topic = topic
        self._process = None
        self._receive_index = 0
        self._latest = None

    async def start(self):
        if self._process is not None:
            return
        if self.topic == "auto":
            self.topic = await discover_configured_topic(TRUTH_CONFIGURED_TOPIC)
        await inspect_topic(self.topic, TRUTH_MESSAGE_TYPE)
        self._process = await asyncio.create_subprocess_exec(
            "gz",
            "topic",
            "-e",
            "--json-output",
            "-t",
            self.topic,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=transport_environment(),
            limit=GAZEBO_JSON_STREAM_LIMIT_BYTES,
        )

    async def events(self, *, timeout_s):
        if self._process is None:
            raise RuntimeError("Gazebo truth source has not been started")
        while True:
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=timeout_s
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError(
                    f"Gazebo truth source timed out after {timeout_s:g}s"
                ) from error
            if not line:
                detail = (await self._process.stderr.read()).decode(
                    errors="replace"
                ).strip()
                raise RuntimeError(
                    "Gazebo truth stream stopped" + (f": {detail}" if detail else "")
                )
            self._receive_index += 1
            try:
                message = json.loads(line)
                truth = parse_gazebo_truth_message(
                    message,
                    topic=self.topic,
                    width=self.width,
                    height=self.height,
                    receive_index=self._receive_index,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                timestamp = 0.0
                try:
                    timestamp = message_timestamp(message)
                except (TypeError, ValueError, UnboundLocalError):
                    pass
                truth = SimulatorTruthFrame(
                    message_id=f"invalid-truth-{self._receive_index:09d}",
                    source_id=f"gazebo_truth:{self.topic}",
                    source_sequence=self._receive_index,
                    sequence_provenance="local_receive_ordinal",
                    simulation_timestamp=timestamp,
                    clock_domain=GAZEBO_SIM_CLOCK,
                    image_width=self.width,
                    image_height=self.height,
                    objects=(),
                    validation_status="invalid",
                    invalid_reasons=(f"{type(error).__name__}: {error}",),
                )
            self._latest = truth
            yield TruthSourceEvent(self._receive_index, truth)

    async def stop(self):
        await stop_stream_process(self._process)
        self._process = None

    def latest(self):
        return self._latest
