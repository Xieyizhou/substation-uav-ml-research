"""Build and validate paired closed-loop comparison matrices."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.scenarios import formal_scenarios


REQUIRED_CONDITIONS = (
    "map_oracle",
    "geometric_lidar",
    "ml_lidar",
    "geometric_ml_fusion",
)


def load_protocol(path: Path):
    with Path(path).open(encoding="utf-8") as source:
        protocol = json.load(source)
    validate_protocol(protocol)
    return protocol


def _scenarios(protocol):
    return protocol.get("scenarios") or formal_scenarios()


def validate_protocol(protocol):
    if set(protocol.get("conditions", [])) != set(REQUIRED_CONDITIONS):
        raise ValueError("protocol must define the four required comparison conditions")
    scenarios = _scenarios(protocol)
    if len(scenarios) != 30:
        raise ValueError("formal protocol requires exactly 30 paired scenarios")
    ids = [row["scenario_id"] for row in scenarios]
    seeds = [int(row["seed"]) for row in scenarios]
    if len(set(ids)) != 30 or set(seeds) != set(range(1001, 1031)):
        raise ValueError("formal scenarios must uniquely use reserved seeds 1001-1030")
    if "extreme" not in {row["map_id"] for row in scenarios}:
        raise ValueError("formal protocol must include the extreme map")
    if not protocol.get("metrics"):
        raise ValueError("protocol must define metrics")


def experiment_matrix(protocol):
    """Expand 30 scenarios only across four paired conditions (120 runs)."""
    validate_protocol(protocol)
    return [
        {**scenario, "condition": condition}
        for scenario in _scenarios(protocol)
        for condition in protocol["conditions"]
    ]
