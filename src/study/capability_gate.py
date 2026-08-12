"""Functional coverage gate required before large flight matrices."""

from __future__ import annotations


QUALIFICATION_CONDITIONS = (
    "geometric_lidar",
    "ml_lidar",
    "geometric_ml_fusion",
)
RUNS_PER_CONDITION = 5


def _sum(rows, metric):
    return sum(float(row.get(metric, 0) or 0) for row in rows)


def _minimum(rows, metric):
    values = [row.get(metric) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return min(float(value) for value in values)


def _condition_coverage(runs, condition):
    selected = [run for run in runs if run.get("condition") == condition]
    completed = [run for run in selected if run.get("status") == "completed"]
    metrics = [run.get("metrics", {}) for run in completed]
    checks = {
        "matrix_complete": len(selected) == RUNS_PER_CONDITION
        and len(completed) == RUNS_PER_CONDITION,
        "mission_success": _sum(metrics, "mission_success") == RUNS_PER_CONDITION,
        "landing_success": _sum(metrics, "landing_success") == RUNS_PER_CONDITION,
        "collision_free": _sum(metrics, "collision_count") == 0,
        "sensor_health": (_minimum(metrics, "sensor_healthy_ratio") or 0) >= 0.95,
        "dynamic_threat_exposure": _sum(
            metrics, "predicted_danger_sample_count"
        ) >= 1,
        "local_replan_attempt": _sum(metrics, "replan_attempt_count") >= 1,
        "local_replan_success": _sum(metrics, "successful_replan_count") >= 1,
        "active_route_replacement": _sum(metrics, "active_replan_count") >= 1,
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


def capability_coverage_report(runs):
    """Report whether every large-study objective was exercised successfully."""
    conditions = {
        condition: _condition_coverage(runs, condition)
        for condition in QUALIFICATION_CONDITIONS
    }
    reasons = [
        f"{condition}: {name} was not demonstrated"
        for condition, report in conditions.items()
        for name, passed in report["checks"].items()
        if not passed
    ]
    return {
        "schema_version": 1,
        "passed": not reasons,
        "required_before_large_study": True,
        "conditions": conditions,
        "reasons": reasons,
    }
