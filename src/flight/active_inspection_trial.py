"""Frozen execution envelope for the first active-inspection flight."""

import asyncio
import hashlib
import json
import math
from pathlib import Path

REQUIRED = {
    "schema_version": 1,
    "trial_id": "active-inspection-first-flight-v1",
    "airborne_budget_s": 90.0,
    "max_semantic_replacements": 2,
    "max_route_distance_m": 8.0,
    "minimum_altitude_m": 1.5,
    "maximum_horizontal_speed_m_s": 1.0,
    "end_behavior": "land_current",
}

QUALIFICATION_REQUIRED = {
    "schema_version": 1,
    "trial_id": "active-inspection-yolo-qualification-v1",
    "airborne_budget_s": 180,
    "max_semantic_replacements": 4,
    "max_route_distance_m": 8,
    "minimum_altitude_m": 1.5,
    "maximum_horizontal_speed_m_s": 1.0,
    "end_behavior": "land_current",
    "exploration_yaw_scan_deg": [0, 90, 180, 270],
    "exploration_yaw_dwell_s": 1.5,
}

FORMAL_REQUIRED = {
    "schema_version": 1,
    "trial_id": "active-inspection-complex-formal-v1",
    "airborne_budget_s": 600,
    "max_semantic_replacements": 10000,
    "max_route_distance_m": 10000,
    "minimum_altitude_m": 1.5,
    "maximum_horizontal_speed_m_s": 1.0,
    "end_behavior": "land_current",
}


def load_active_inspection_trial(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload not in (REQUIRED, QUALIFICATION_REQUIRED, FORMAL_REQUIRED):
        raise ValueError("active inspection trial configuration is not frozen v1")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "artifact_identity": hashlib.sha256(encoded).hexdigest()}


def truncate_trial_route(waypoints, maximum_distance_m, minimum_altitude_m):
    result = []
    distance = 0.0
    previous = None
    truncated = False
    for waypoint in waypoints:
        current = dict(waypoint)
        if "down_m" in current:
            current["down_m"] = -max(
                float(minimum_altitude_m), abs(float(current["down_m"]))
            )
        else:
            current["altitude_m"] = max(
                float(minimum_altitude_m), float(current.get("altitude_m", 0))
            )
        if previous is not None:
            segment = math.hypot(
                current["east_m"] - previous["east_m"],
                current["north_m"] - previous["north_m"],
            )
            if distance + segment > maximum_distance_m:
                truncated = True
                break
            distance += segment
        result.append(current)
        previous = current
    if len(result) < len(waypoints):
        truncated = True
    return result, truncated, distance


async def wait_for_trial_budget(latest, budget_s):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 120.0
    while not bool(latest.get("in_air")):
        if loop.time() >= deadline:
            raise TimeoutError("in-air telemetry did not become truthy within 120 seconds")
        await asyncio.sleep(0.1)
    await asyncio.sleep(float(budget_s))
