"""Value parsing and CSV-derived metrics for experiment summaries."""

import csv
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A final position error below this value is treated as good enough for a
# high-level benchmark pass. The low-level waypoint acceptance threshold is
# still recorded by the flight analysis summaries.
FINAL_ERROR_PASS_THRESHOLD_M = 1.0

SUMMARY_COLUMNS = [
    "status",
    "run_id",
    "source_log",
    "risk_action",
    "map_name",
    "altitude_m",
    "max_speed_m_s",
    "speed_source",
    "return_home_enabled",
    "duration_s",
    "warning_sample_ratio",
    "danger_sample_ratio",
    "percent_time_warning",
    "percent_time_danger",
    "minimum_nearest_obstacle_distance_m",
    "mean_nearest_obstacle_distance_m",
    "median_nearest_obstacle_distance_m",
    "max_horizontal_error_m",
    "mean_horizontal_error_m",
    "final_horizontal_error_m",
    "approximate_min_clearance_m",
    "raw_physical_collision_detected",
    "inflated_safety_buffer_entry_detected",
    "obstacle_collision_detected",
    "active_route_replacement_count",
    "max_active_replan_path_length",
    "waypoints_reached",
]


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        print(f"WARNING: Could not parse {path}: {error}")
        return {}


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def clean_text(value):
    if value is None:
        return None
    return str(value).strip().strip("`")


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def parse_float(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def parse_bool(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().strip("`").lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def format_float(value):
    value = parse_float(value)
    if value is None:
        return ""
    return f"{value:.3f}"


def format_bool(value):
    parsed = parse_bool(value)
    if parsed is None:
        return ""
    return "yes" if parsed else "no"


def parse_summary_bullets(summary_text):
    values = {}
    for line in summary_text.splitlines():
        match = re.match(r"^- ([^:]+):\s*(.*)$", line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower().replace("-", "_").replace(" ", "_")
        values[key] = clean_text(match.group(2))
    return values


def waypoint_reached_count(summary_text):
    count = 0
    for line in summary_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        if cells[0] in {"route_direction", "---"}:
            continue
        if cells[5].lower() == "yes":
            count += 1
    return count if count else None


def resolve_project_path(path_text):
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def csv_rows(source_log):
    path = resolve_project_path(source_log)
    if path is None or not path.exists():
        return []
    try:
        with path.open(newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except OSError as error:
        print(f"WARNING: Could not read {path}: {error}")
        return []


def first_csv_value(rows, column):
    for row in rows:
        value = row.get(column)
        if value not in {None, ""}:
            return value
    return None


def last_csv_float(rows, column):
    for row in reversed(rows):
        value = parse_float(row.get(column))
        if value is not None:
            return value
    return None


def max_csv_float(rows, column):
    values = [parse_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def count_csv_true(rows, column):
    if not rows or not any(column in row for row in rows):
        return None
    return sum(1 for row in rows if parse_bool(row.get(column)))


def mean_csv_float(rows, column):
    values = [parse_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def warning_note(warnings):
    if not warnings:
        return ""
    if isinstance(warnings, list):
        return "; ".join(str(warning) for warning in warnings if warning)
    return str(warnings)


def target_points(rows):
    points = []
    previous_key = None
    for row in rows:
        name = str(row.get("target_name") or "").strip()
        north = parse_float(row.get("target_north_m"))
        east = parse_float(row.get("target_east_m"))
        if not name or north is None or east is None:
            continue
        key = (row.get("route_direction") or "", name, north, east)
        if key == previous_key:
            continue
        points.append((north, east))
        previous_key = key
    return points


def path_length_from_points(points):
    total = 0.0
    previous = None
    for point in points:
        if previous is not None:
            total += math.hypot(point[0] - previous[0], point[1] - previous[1])
        previous = point
    return total if total > 0 else None


def planned_path_length(rows):
    return path_length_from_points(target_points(rows))


def approximate_distance_traveled(rows):
    total = 0.0
    previous = None
    for row in rows:
        north = parse_float(row.get("local_north_m"))
        east = parse_float(row.get("local_east_m"))
        if north is None or east is None:
            continue
        current = (north, east)
        if previous is not None:
            step = math.hypot(current[0] - previous[0], current[1] - previous[1])
            if step <= 3.0:
                total += step
        previous = current
    return total if total > 0 else None


def count_risk_rows(rows, risk_levels=None):
    if not rows or not any("perception_risk_level" in row for row in rows):
        return None
    risk_levels = set(risk_levels or {"detected", "warning", "danger"})
    return sum(
        1
        for row in rows
        if str(row.get("perception_risk_level") or "").strip().lower() in risk_levels
    )


def slow_down_event_count(rows, risk_action):
    if risk_action != "slow_down":
        return 0 if risk_action else None
    count = count_risk_rows(rows, {"warning", "danger"})
    return count


def completed_or_failed(status):
    if status == "FAIL":
        return "failed"
    if status == "PASS":
        return "completed"
    return "needs_review"


def measured_max_horizontal_speed(rows):
    speeds = []
    for row in rows:
        north = parse_float(row.get("velocity_north_m_s"))
        east = parse_float(row.get("velocity_east_m_s"))
        if north is None or east is None:
            continue
        speeds.append(math.hypot(north, east))
    return max(speeds) if speeds else None


def commanded_speed(rows):
    speed_columns = [
        "commanded_speed_m_s",
        "commanded_max_speed_m_s",
        "max_speed_m_s",
        "max_horizontal_speed_m_s",
    ]
    for column in speed_columns:
        value = max_csv_float(rows, column)
        if value is not None:
            return value, column
    return None, None


def choose_status(raw_collision, inflated_entry, final_error, run_status=None):
    if run_status:
        if (
            run_status.get("status") != "completed"
            or run_status.get("landing_confirmed") is not True
        ):
            return "FAIL"
    if parse_bool(raw_collision):
        return "FAIL"
    if parse_bool(inflated_entry):
        return "WARNING"
    final_error = parse_float(final_error)
    if final_error is None or final_error > FINAL_ERROR_PASS_THRESHOLD_M:
        return "WARNING"
    return "PASS"
