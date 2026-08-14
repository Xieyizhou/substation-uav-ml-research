"""Functional coverage gate required before large flight matrices."""

from __future__ import annotations

from src.study.challenge_spec import load_challenge_spec


QUALIFICATION_CONDITIONS = (
    "geometric_lidar",
    "ml_lidar",
    "geometric_ml_fusion",
)
RUNS_PER_CONDITION = {
    "challenge": 1,
    "closed-loop": 5,
    "formal": 30,
}
DEFAULT_ACCEPTANCE = {
    "active_route_replacement_min": 1,
    "collision_count_max": 0,
    "landing_success_required": True,
    "mission_success_required": True,
    "predicted_danger_sample_count_min": 1,
    "replan_attempt_count_min": 1,
    "sensor_healthy_ratio_min": 0.95,
    "successful_replan_count_min": 1,
}


def _sum(rows, metric):
    return sum(float(row.get(metric, 0) or 0) for row in rows)


def _minimum(rows, metric):
    values = [row.get(metric) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return min(float(value) for value in values)


def _condition_coverage(runs, condition, expected_count, acceptance):
    selected = [run for run in runs if run.get("condition") == condition]
    completed = [run for run in selected if run.get("status") == "completed"]
    metrics = [run.get("metrics", {}) for run in completed]
    checks = {
        "matrix_complete": len(selected) == expected_count
        and len(completed) == expected_count,
        "mission_success": not acceptance["mission_success_required"]
        or _sum(metrics, "mission_success") == expected_count,
        "landing_success": not acceptance["landing_success_required"]
        or _sum(metrics, "landing_success") == expected_count,
        "collision_free": _sum(metrics, "collision_count")
        <= acceptance["collision_count_max"],
        "sensor_health": (_minimum(metrics, "sensor_healthy_ratio") or 0)
        >= acceptance["sensor_healthy_ratio_min"],
        "dynamic_threat_exposure": _sum(
            metrics, "predicted_danger_sample_count"
        ) >= acceptance["predicted_danger_sample_count_min"],
        "local_replan_attempt": _sum(metrics, "replan_attempt_count")
        >= acceptance["replan_attempt_count_min"],
        "local_replan_success": _sum(metrics, "successful_replan_count")
        >= acceptance["successful_replan_count_min"],
        "active_route_replacement": _sum(metrics, "active_replan_count")
        >= acceptance["active_route_replacement_min"],
    }
    return {
        "passed": all(checks.values()),
        "scheduled": len(selected),
        "completed": len(completed),
        "checks": checks,
        "observed": {
            "predicted_danger_sample_count": int(
                _sum(metrics, "predicted_danger_sample_count")
            ),
            "truth_danger_sample_count": int(
                _sum(metrics, "truth_danger_sample_count")
            ),
            "replan_attempt_count": int(_sum(metrics, "replan_attempt_count")),
            "successful_replan_count": int(
                _sum(metrics, "successful_replan_count")
            ),
            "active_replan_count": int(_sum(metrics, "active_replan_count")),
            "minimum_sensor_healthy_ratio": _minimum(
                metrics, "sensor_healthy_ratio"
            ),
        },
    }


def capability_coverage_report(runs, *, tier="closed-loop"):
    """Report whether every large-study objective was exercised successfully."""
    if tier not in RUNS_PER_CONDITION:
        raise ValueError(f"unsupported capability tier: {tier}")
    expected_count = RUNS_PER_CONDITION[tier]
    acceptance = (
        load_challenge_spec()["acceptance"]
        if tier == "challenge"
        else DEFAULT_ACCEPTANCE
    )
    conditions = {
        condition: _condition_coverage(
            runs, condition, expected_count, acceptance
        )
        for condition in QUALIFICATION_CONDITIONS
    }
    reasons = [
        f"{condition}: {name} was not demonstrated"
        for condition, report in conditions.items()
        for name, passed in report["checks"].items()
        if not passed
    ]
    return {
        "schema_version": 2,
        "tier": tier,
        "passed": not reasons,
        "required_before_large_study": tier in {"challenge", "closed-loop"},
        "conditions": conditions,
        "reasons": reasons,
    }
