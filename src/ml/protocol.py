"""Build and validate the formal closed-loop comparison matrix."""

from __future__ import annotations

import json
from pathlib import Path


def load_protocol(path: Path):
    with Path(path).open(encoding="utf-8") as source:
        protocol = json.load(source)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    required_conditions = {
        "map_oracle",
        "geometric_lidar",
        "ml_lidar",
        "geometric_ml_fusion",
    }
    conditions = set(protocol.get("conditions", []))
    if conditions != required_conditions:
        raise ValueError("protocol must define the four required comparison conditions")
    seeds = protocol.get("seeds", [])
    if len(seeds) < 30 or len(set(seeds)) != len(seeds):
        raise ValueError("formal protocol requires at least 30 unique seeds")
    if "extreme" not in protocol.get("maps", []):
        raise ValueError("formal protocol must include the extreme map")
    if not protocol.get("metrics"):
        raise ValueError("protocol must define metrics")


def experiment_matrix(protocol):
    validate_protocol(protocol)
    return [
        {
            "map_id": map_id,
            "target_id": target,
            "condition": condition,
            "seed": seed,
        }
        for map_id in protocol["maps"]
        for target in protocol["targets"]
        for condition in protocol["conditions"]
        for seed in protocol["seeds"]
    ]
