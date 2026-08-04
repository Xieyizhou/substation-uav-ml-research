"""Registry and validation boundary for visual collection protocols."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256
from src.maps.map_catalog import PROJECT_ROOT


V1_PROTOCOL_ID = "visual-multiscenario-png-v1"
V2_PROTOCOL_ID = "visual-multiscenario-png-v2"
PROTOCOL_PATHS = {
    V1_PROTOCOL_ID: PROJECT_ROOT / "benchmarks/visual_static_v1/collection_protocol.json",
    V2_PROTOCOL_ID: PROJECT_ROOT / "benchmarks/visual_static_v2/collection_protocol.json",
}
PROTOCOL_ALIASES = {"v1": V1_PROTOCOL_ID, "v2": V2_PROTOCOL_ID, **{key: key for key in PROTOCOL_PATHS}}


@dataclass(frozen=True)
class CollectionProtocol(Mapping):
    protocol_id: str
    path: Path
    data: dict

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    @property
    def identity_sha256(self):
        return object_sha256(self.data)


def resolve_protocol_path(reference=None):
    if reference is None:
        return PROTOCOL_PATHS[V1_PROTOCOL_ID]
    text = str(reference)
    protocol_id = PROTOCOL_ALIASES.get(text)
    return PROTOCOL_PATHS[protocol_id] if protocol_id else Path(reference)


def protocol_path_for_id(protocol_id):
    try:
        return PROTOCOL_PATHS[str(protocol_id)]
    except KeyError as error:
        raise ValueError(f"unsupported visual collection protocol: {protocol_id}") from error


def _read_protocol(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing visual collection file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed visual collection JSON: {path}: {error}") from error


def _validate_common(data):
    if data.get("canonical_payload_format") != "png" or data.get("canonical_pixel_format") != "rgb8":
        raise ValueError("visual collection protocols require canonical PNG RGB8")
    required = tuple(data.get("aggregate_gate", {}).get("required_classes_per_split", []))
    if required != tuple(EQUIPMENT_CLASSES):
        raise ValueError("visual collection class order is not locked")
    recording = data.get("recording") or {}
    if (
        recording.get("recordings_per_scenario") != 1
        or not recording.get("retain_every_valid_source_frame")
        or recording.get("automatic_downsampling") is not False
        or recording.get("default_phase_source") != "flight_lifecycle"
        or recording.get("automatic_stop_condition") != "completed_landed_and_landing_confirmed"
    ):
        raise ValueError("visual collection recording policy is invalid")


def _validate_v1(data):
    if data.get("collection_protocol_schema_version") != 1:
        raise ValueError("unsupported v1 visual collection schema")
    if set(data.get("scenario_allocation") or {}) != {"train", "validation", "test"}:
        raise ValueError("v1 collection must define train, validation, and test")
    randomization = data.get("randomization") or {}
    applied = tuple(randomization.get("applied_visual_fields", []))
    expected = ("equipment_scale", "equipment_position_jitter_m", "light_intensity", "unknown_obstacles")
    if applied != expected:
        raise ValueError("v1 visual randomization is not frozen")
    if set(applied).intersection(randomization.get("explicitly_not_applied", [])):
        raise ValueError("visual randomization fields cannot be applied and excluded")


def _validate_v2(data):
    if data.get("collection_protocol_schema_version") != 2:
        raise ValueError("unsupported v2 visual collection schema")
    allocation = data.get("layout_allocation") or {}
    if tuple(allocation) != ("development", "validation", "blind"):
        raise ValueError("v2 collection must define development, validation, and blind layouts")
    layout_ids = []
    seeds = []
    for split, expected_count in (("development", 6), ("validation", 2), ("blind", 2)):
        spec = allocation[split]
        ids = tuple(spec.get("layout_ids", []))
        split_seeds = list(range(int(spec["seed_first"]), int(spec["seed_last"]) + 1))
        if len(ids) != expected_count or len(split_seeds) != expected_count * 5:
            raise ValueError(f"v2 {split} layout allocation is invalid")
        layout_ids.extend(ids)
        seeds.extend(split_seeds)
    if len(set(layout_ids)) != 10 or seeds != list(range(3001, 3051)):
        raise ValueError("v2 layout IDs and seeds are not frozen")
    routes = data.get("routes") or []
    route_classes = tuple(route.get("target_class") for route in routes)
    if route_classes != (*EQUIPMENT_CLASSES, None):
        raise ValueError("v2 routes must cover four classes then background")
    if data.get("split_policy", {}).get("minimum_split_unit") != "layout":
        raise ValueError("v2 minimum split unit must be layout")
    if data.get("randomization", {}).get("generator_version") != "gazebo-sdf-v3":
        raise ValueError("v2 requires gazebo-sdf-v3")
    timing = data.get("flight_timeout_policy") or {}
    if (
        float(timing.get("nominal_horizontal_speed_m_s", 0.0)) <= 0.0
        or float(timing.get("timeout_multiplier", 0.0)) < 1.0
        or float(timing.get("minimum_timeout_s", 0.0)) <= 0.0
        or float(timing.get("maximum_timeout_s", 0.0))
        < float(timing.get("minimum_timeout_s", 0.0))
    ):
        raise ValueError("v2 flight timeout policy is invalid")


VALIDATORS = {V1_PROTOCOL_ID: _validate_v1, V2_PROTOCOL_ID: _validate_v2}


def load_protocol(reference=None):
    path = resolve_protocol_path(reference)
    data = _read_protocol(path)
    protocol_id = data.get("protocol_id")
    validator = VALIDATORS.get(protocol_id)
    if validator is None:
        raise ValueError("unsupported visual collection protocol")
    _validate_common(data)
    validator(data)
    return CollectionProtocol(protocol_id, path, data)


def protocol_for_plan(plan):
    protocol = load_protocol(protocol_path_for_id(plan.get("protocol_id")))
    supplied = plan.get("protocol_identity_sha256")
    if supplied is not None and supplied != protocol.identity_sha256:
        raise ValueError("collection protocol identity mismatch")
    return protocol
