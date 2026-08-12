"""Canonical data splits and balanced research scenario plans."""

from __future__ import annotations

TRAINING_MAPS = ("training", "simple", "medium", "complex")
EVALUATION_SEEDS = frozenset(range(1001, 1031))
SPLIT_SEEDS = {
    "train": frozenset(range(2001, 2041)),
    "validation": frozenset(range(2041, 2051)),
    "test": frozenset(range(2051, 2061)),
}
TARGETS = ("top_right", "center", "left", "bottom", "right")


def split_for(map_id: str, seed: int) -> str:
    if seed in EVALUATION_SEEDS:
        raise ValueError("formal evaluation seeds 1001-1030 cannot enter a dataset")
    if seed in SPLIT_SEEDS["train"]:
        if map_id not in TRAINING_MAPS:
            raise ValueError("training split excludes the held-out extreme map")
        return "train"
    if seed in SPLIT_SEEDS["validation"]:
        if map_id not in TRAINING_MAPS:
            raise ValueError("validation split excludes the held-out extreme map")
        return "validation"
    if seed in SPLIT_SEEDS["test"]:
        if map_id != "extreme":
            raise ValueError("test split is reserved for the extreme map")
        return "test"
    raise ValueError("seed is outside the versioned dataset split ranges")


def scenario_id(map_id: str, target_id: str, seed: int) -> str:
    return f"{map_id}-{target_id}-{int(seed)}"


def formal_scenarios():
    """Thirty balanced, paired scenarios; conditions are added separately."""
    maps = ("simple", "medium", "complex", "extreme")
    return [
        {
            "scenario_id": scenario_id(
                maps[index % len(maps)], TARGETS[index % len(TARGETS)], seed
            ),
            "map_id": maps[index % len(maps)],
            "target_id": TARGETS[index % len(TARGETS)],
            "seed": seed,
        }
        for index, seed in enumerate(range(1001, 1031))
    ]


def closed_loop_scenarios():
    maps = ("simple", "medium", "complex", "extreme", "complex")
    scenarios = [
        {
            "scenario_id": scenario_id(map_id, target_id, seed),
            "map_id": map_id,
            "target_id": target_id,
            "seed": seed,
        }
        for map_id, target_id, seed in zip(maps, TARGETS, range(1001, 1006))
    ]
    scenarios[0]["scenario_profile"] = "unmapped_route_blocker_v1"
    scenarios[0]["required_capabilities"] = (
        "dynamic_threat_detection",
        "local_replan_attempt",
        "local_replan_success",
        "active_route_replacement",
        "safe_mission_completion",
    )
    return scenarios
