"""Frozen speed matrix and conservative recommendation logic."""

from __future__ import annotations

import json
from math import ceil, isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "config/planning/speed_envelope_v1.json"


def load_speed_spec(path=DEFAULT_SPEC):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("speed_envelope_schema_version") != 1:
        raise ValueError("unsupported speed envelope schema")
    if len(value.get("representative_scenarios", [])) != 4:
        raise ValueError("speed envelope requires four representative scenarios")
    if value.get("speeds_m_s") != [0.5, 0.75, 1.0, 1.25, 1.5]:
        raise ValueError("speed envelope requires the frozen five-speed sequence")
    if value.get("repeats") != 3:
        raise ValueError("speed envelope requires three repeats")
    return value


def speed_envelope_matrix(path=DEFAULT_SPEC):
    spec = load_speed_spec(path)
    rows = []
    index = 0
    for scenario in spec["representative_scenarios"]:
        for speed in spec["speeds_m_s"]:
            for repeat in range(1, spec["repeats"] + 1):
                index += 1
                speed_token = str(speed).replace(".", "p")
                rows.append({
                    "scenario_id": (
                        f"speed-{scenario['id']}-{speed_token}-r{repeat:02d}"
                    ),
                    "map_id": scenario["map_id"],
                    "target_id": "top_right",
                    "seed": spec["seed"] + index - 1,
                    "condition": "geometric_lidar",
                    "injection_phase": scenario["injection_phase"],
                    "representative_scenario": scenario["id"],
                    "speed_m_s": float(speed),
                    "repeat": repeat,
                })
    return rows


def _metrics(row):
    return row.get("metrics", row)


def _speed_report(rows, speed, limits, expected_count):
    selected = [
        row for row in rows
        if isclose(
            float(_metrics(row).get("configured_speed_m_s", -1)), speed,
            rel_tol=0.0, abs_tol=1e-9,
        )
    ]
    reasons = []
    if len(selected) != expected_count:
        reasons.append(f"expected {expected_count} runs, found {len(selected)}")
    count = len(selected)
    values = [_metrics(row) for row in selected]
    rate = lambda name: (
        sum(float(value.get(name, 0)) for value in values) / count if count else 0.0
    )
    total = lambda name: sum(float(value.get(name, 0)) for value in values)
    latencies = sorted(
        float(value["detection_to_resume_ms"])
        for value in values
        if value.get("detection_to_resume_ms") is not None
    )
    p95_index = max(0, ceil(len(latencies) * 0.95) - 1)
    metrics = {
        "mission_completion_rate": rate("mission_success"),
        "route_switch_correctness": rate("route_switch_correct"),
        "landing_success_rate": rate("landing_success"),
        "event_chain_completion_rate": rate("event_chain_complete"),
        "collision_count": total("collision_count"),
        "safety_failure_count": total("safety_failure_count"),
        "detection_to_resume_p95_ms": (
            latencies[p95_index] if len(latencies) == count and count else None
        ),
    }
    for name in (
        "mission_completion_rate",
        "route_switch_correctness",
        "landing_success_rate",
        "event_chain_completion_rate",
    ):
        if metrics[name] < limits[f"{name}_min"]:
            reasons.append(f"{name} below minimum")
    for name in ("collision_count", "safety_failure_count"):
        if metrics[name] > limits[f"{name}_max"]:
            reasons.append(f"{name} above maximum")
    latency = metrics["detection_to_resume_p95_ms"]
    if latency is None:
        reasons.append("detection-to-resume latency evidence is incomplete")
    elif latency > limits["detection_to_resume_p95_ms_max"]:
        reasons.append("detection_to_resume_p95_ms above maximum")
    return {
        "speed_m_s": speed,
        "passed": not reasons,
        "reasons": reasons,
        "run_count": count,
        "metrics": metrics,
    }


def speed_envelope_report(rows, spec_path=DEFAULT_SPEC):
    spec = load_speed_spec(spec_path)
    expected = len(spec["representative_scenarios"]) * spec["repeats"]
    reports = [
        _speed_report(rows, float(speed), spec["acceptance"], expected)
        for speed in spec["speeds_m_s"]
    ]
    contiguous_passing = []
    for report in reports:
        if not report["passed"]:
            break
        contiguous_passing.append(report["speed_m_s"])
    expected_total = expected * len(spec["speeds_m_s"])
    observed_total = sum(report["run_count"] for report in reports)
    complete = observed_total == expected_total and all(
        report["run_count"] == expected for report in reports
    )
    return {
        "passed": complete and bool(contiguous_passing),
        "recommended_max_speed_m_s": (
            max(contiguous_passing) if complete and contiguous_passing else None
        ),
        "expected_run_count": expected_total,
        "observed_run_count": observed_total,
        "speed_reports": reports,
    }
