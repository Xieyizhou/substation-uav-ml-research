"""Paired comparisons, confidence intervals, and promotion gates."""

from __future__ import annotations

import random

from src.ml.metrics import percentile


LOWER_IS_BETTER = {
    "collision_count",
    "near_miss_count",
    "buffer_entry_count",
    "path_length_m",
    "flight_time_s",
    "replan_p95_ms",
    "inference_p95_ms",
    "risk_ece",
    "risk_false_negative_rate",
    "direction_mae_deg",
}

FORMAL_COMPARISONS = (
    ("ml_lidar", "geometric_lidar"),
    ("geometric_ml_fusion", "geometric_lidar"),
    ("geometric_ml_fusion", "ml_lidar"),
    ("map_oracle", "geometric_lidar"),
)


def paired_differences(runs, metric, candidate, baseline):
    grouped = {}
    for run in runs:
        value = run.get("metrics", {}).get(metric)
        if value is not None:
            grouped.setdefault(run["scenario_id"], {})[run["condition"]] = float(value)
    differences = []
    for values in grouped.values():
        if candidate in values and baseline in values:
            difference = values[candidate] - values[baseline]
            if metric in LOWER_IS_BETTER:
                difference *= -1
            differences.append(difference)
    return differences


def bootstrap_interval(values, *, confidence=0.95, samples=2000, seed=17):
    values = [float(value) for value in values]
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "count": 0}
    generator = random.Random(seed)
    means = [
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": sum(values) / len(values),
        "ci_low": percentile(means, alpha),
        "ci_high": percentile(means, 1.0 - alpha),
        "count": len(values),
    }


def comparison_report(runs):
    metrics = sorted(
        {
            name
            for run in runs
            for name in run.get("metrics", {})
        }
    )
    report = {}
    for metric in metrics:
        candidate = (
            "ml_lidar"
            if any(run["condition"] == "ml_lidar" for run in runs)
            else "candidate"
        )
        baseline = (
            "geometric_lidar"
            if any(run["condition"] == "geometric_lidar" for run in runs)
            else "champion"
        )
        report[metric] = bootstrap_interval(
            paired_differences(runs, metric, candidate, baseline)
        )
    return report


def _descriptive_report(runs):
    conditions = sorted({run["condition"] for run in runs})
    report = {}
    for condition in conditions:
        selected = [run for run in runs if run["condition"] == condition]
        completed = [run for run in selected if run["status"] == "completed"]
        metrics = sorted({name for run in completed for name in run.get("metrics", {})})
        report[condition] = {
            "scheduled": len(selected),
            "completed": len(completed),
            "failed_or_missing": len(selected) - len(completed),
            "metrics": {
                name: sum(
                    float(run["metrics"][name])
                    for run in completed if name in run.get("metrics", {})
                ) / max(sum(name in run.get("metrics", {}) for run in completed), 1)
                for name in metrics
            },
        }
    return report


def formal_comparison_report(runs, *, samples=2000, seed=17):
    """Build the frozen four-condition paired formal comparison report."""
    comparisons = {}
    all_metrics = sorted({name for run in runs for name in run.get("metrics", {})})
    for candidate, baseline in FORMAL_COMPARISONS:
        metrics = {}
        for name in all_metrics:
            differences = paired_differences(runs, name, candidate, baseline)
            if differences:
                metrics[name] = bootstrap_interval(
                    differences, samples=samples, seed=seed
                )
        comparisons[f"{candidate}__vs__{baseline}"] = {
            "candidate": candidate,
            "baseline": baseline,
            "metrics": metrics,
        }
    return {
        "formal_comparison_schema_version": 1,
        "bootstrap": {"samples": samples, "seed": seed, "confidence": 0.95},
        "descriptive": _descriptive_report(runs),
        "paired_comparisons": comparisons,
    }


def replay_gate(candidate, champion=None):
    latency = candidate.get("inference_p95_ms")
    danger = candidate.get("danger_recall")
    reasons = []
    if latency is None or latency > 50.0:
        reasons.append("ONNX CPU P95 latency exceeds 50 ms or is missing")
    if champion and danger is not None and champion.get("danger_recall") is not None:
        if danger < champion["danger_recall"] - 0.02:
            reasons.append("danger recall regressed by more than 2 percentage points")
    return {"passed": not reasons, "reasons": reasons}


def closed_loop_gate(runs):
    incomplete = [run for run in runs if run["status"] != "completed"]
    if incomplete:
        return {
            "passed": False,
            "reasons": [f"{len(incomplete)} closed-loop run(s) are incomplete"],
        }
    by_condition = {}
    for run in runs:
        by_condition.setdefault(run["condition"], []).append(run.get("metrics", {}))
    candidate = by_condition.get("ml_lidar", [])
    baseline = by_condition.get("geometric_lidar", [])
    reasons = []
    if not candidate or not baseline:
        reasons.append("paired geometric and ML closed-loop results are missing")
    for unsafe in ("collision_count", "safety_failure_count"):
        if sum(row.get(unsafe, 0) for row in candidate) > sum(
            row.get(unsafe, 0) for row in baseline
        ):
            reasons.append(f"{unsafe} regressed")
    return {"passed": not reasons, "reasons": reasons}


def promotion_gate(runs, *, minimum_quality_gain=0.005, minimum_latency_gain=0.01):
    by_condition = {}
    for run in runs:
        if run["status"] != "completed":
            continue
        by_condition.setdefault(run["condition"], []).append(run.get("metrics", {}))
    candidate = by_condition.get("ml_lidar", [])
    baseline = by_condition.get("geometric_lidar", [])
    if not candidate or not baseline:
        return {"passed": False, "reasons": ["paired formal results are incomplete"]}
    for unsafe in ("collision_count", "safety_failure_count"):
        if sum(row.get(unsafe, 0) for row in candidate) > sum(
            row.get(unsafe, 0) for row in baseline
        ):
            return {"passed": False, "reasons": [f"{unsafe} regressed"]}
    improvements = []
    for metric in ("risk_f1", "traversability_iou"):
        candidate_mean = sum(row.get(metric, 0) for row in candidate) / len(candidate)
        baseline_mean = sum(row.get(metric, 0) for row in baseline) / len(baseline)
        improvements.append(candidate_mean - baseline_mean >= minimum_quality_gain)
    candidate_latency = sum(
        row.get("inference_p95_ms", float("inf")) for row in candidate
    ) / len(candidate)
    baseline_latency = sum(
        row.get("inference_p95_ms", float("inf")) for row in baseline
    ) / len(baseline)
    improvements.append(
        baseline_latency not in (0.0, float("inf"))
        and (baseline_latency - candidate_latency) / baseline_latency
        >= minimum_latency_gain
    )
    return {
        "passed": any(improvements),
        "reasons": [] if any(improvements) else ["no quality or latency metric improved"],
    }


def _condition_mean(runs, condition):
    selected = [
        run.get("metrics", {})
        for run in runs
        if run["condition"] == condition and run["status"] == "completed"
    ]
    names = {name for row in selected for name in row}
    return {
        name: sum(float(row[name]) for row in selected if name in row)
        / sum(name in row for row in selected)
        for name in names
    }


def study_gate_report(runs_by_tier):
    replay = runs_by_tier.get("replay", [])
    replay_incomplete = [run for run in replay if run["status"] != "completed"]
    replay_result = (
        {
            "passed": False,
            "reasons": [f"{len(replay_incomplete)} replay run(s) are incomplete"],
        }
        if replay_incomplete or not replay
        else replay_gate(
            _condition_mean(replay, "candidate"),
            _condition_mean(replay, "champion") or None,
        )
    )
    closed_result = closed_loop_gate(runs_by_tier.get("closed-loop", []))
    formal = runs_by_tier.get("formal", [])
    formal_result = (
        {
            "passed": False,
            "reasons": [
                f"{sum(run['status'] != 'completed' for run in formal)} "
                "formal run(s) are incomplete"
            ],
        }
        if not formal or any(run["status"] != "completed" for run in formal)
        else promotion_gate(formal)
    )
    return {
        "passed": all(
            result["passed"]
            for result in (replay_result, closed_result, formal_result)
        ),
        "replay": replay_result,
        "closed_loop": closed_result,
        "formal": formal_result,
    }
