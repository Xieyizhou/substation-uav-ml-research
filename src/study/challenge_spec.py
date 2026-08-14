"""Versioned, deterministic capability challenge specification."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHALLENGE_SPEC = ROOT / "config/perception/lidar_challenge_gate_v1.json"
CHALLENGE_CONDITIONS = (
    "geometric_lidar",
    "ml_lidar",
    "geometric_ml_fusion",
)
ACCEPTANCE_FIELDS = frozenset({
    "active_route_replacement_min",
    "collision_count_max",
    "landing_success_required",
    "mission_success_required",
    "predicted_danger_sample_count_min",
    "replan_attempt_count_min",
    "sensor_healthy_ratio_min",
    "successful_replan_count_min",
})


def load_challenge_spec(path=DEFAULT_CHALLENGE_SPEC):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("challenge_gate_schema_version") != 1:
        raise ValueError("unsupported challenge gate schema")
    if tuple(value.get("conditions", ())) != CHALLENGE_CONDITIONS:
        raise ValueError("challenge conditions do not match the supported matrix")
    if set(value.get("acceptance", {})) != ACCEPTANCE_FIELDS:
        raise ValueError("challenge acceptance contract is incomplete")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("challenge gate requires at least one scenario")
    identities = {item.get("scenario_id") for item in scenarios}
    if None in identities or len(identities) != len(scenarios):
        raise ValueError("challenge scenario identities must be unique")
    for scenario in scenarios:
        required = ("map_id", "target_id", "seed", "scenario_profile")
        if any(name not in scenario for name in required):
            raise ValueError("challenge scenario is incomplete")
    return value


def challenge_matrix(path=DEFAULT_CHALLENGE_SPEC):
    specification = load_challenge_spec(path)
    return [
        {**scenario, "condition": condition}
        for scenario in specification["scenarios"]
        for condition in specification["conditions"]
    ]
