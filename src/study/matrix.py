"""Tier-specific, paired study matrices."""

from __future__ import annotations

from src.ml.scenarios import closed_loop_scenarios, formal_scenarios
from src.study.challenge_spec import challenge_matrix
from src.study.dynamic_replanning import dynamic_replanning_matrix
from src.study.speed_envelope import speed_envelope_matrix


FORMAL_CONDITIONS = (
    "map_oracle",
    "geometric_lidar",
    "ml_lidar",
    "geometric_ml_fusion",
)
CLOSED_LOOP_CONDITIONS = (
    "geometric_lidar",
    "ml_lidar",
    "geometric_ml_fusion",
)


def tier_matrix(tier, *, include_champion=True):
    if tier == "speed-envelope":
        return speed_envelope_matrix()
    if tier == "dynamic-replanning":
        return dynamic_replanning_matrix()
    if tier == "replay":
        conditions = ("champion", "candidate") if include_champion else ("candidate",)
        return [
            {
                "scenario_id": f"replay-{index:02d}",
                "map_id": map_id,
                "target_id": "recorded",
                "seed": seed,
                "condition": condition,
            }
            for index, (map_id, seed) in enumerate(
                zip(("simple", "medium", "complex", "extreme", "extreme"), range(1001, 1006)),
                start=1,
            )
            for condition in conditions
        ]
    if tier == "closed-loop":
        return [
            {**scenario, "condition": condition}
            for scenario in closed_loop_scenarios()
            for condition in CLOSED_LOOP_CONDITIONS
        ]
    if tier == "challenge":
        return challenge_matrix()
    if tier == "formal":
        return [
            {**scenario, "condition": condition}
            for scenario in formal_scenarios()
            for condition in FORMAL_CONDITIONS
        ]
    raise ValueError(f"unsupported study tier: {tier}")
