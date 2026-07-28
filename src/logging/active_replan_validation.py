"""Offline validation of active-replan outbound target switching."""

import math
import re


WP_PATTERN = re.compile(r"^WP\d+$")
RWP_PATTERN = re.compile(r"^RWP(\d+)$")
DEFAULT_HORIZONTAL_ACCEPTANCE_M = 0.4
DEFAULT_VERTICAL_ACCEPTANCE_M = 0.4


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _bool(value):
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "on"}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _same_target_coordinates(row, goal, tolerance=1e-6):
    for key in ("target_north_m", "target_east_m", "target_down_m"):
        value = _number(row.get(key))
        goal_value = _number(goal.get(key))
        if value is None or goal_value is None or abs(value - goal_value) > tolerance:
            return False
    return True


def _replacement_index(rows):
    """Return the first outbound replacement event and its signal availability."""
    has_flag = any("replan_route_replaced" in row for row in rows)
    has_count = any("active_replan_count" in row for row in rows)
    previous_count = 0.0
    for index, row in enumerate(rows):
        count = _number(row.get("active_replan_count"))
        count_increased = count is not None and count > previous_count
        if count is not None:
            previous_count = max(previous_count, count)
        if _text(row.get("route_direction")).lower() != "outbound":
            continue
        if _bool(row.get("replan_route_replaced")) or count_increased:
            return index, has_flag or has_count
    return None, has_flag or has_count


def validate_active_replan_rows(
    rows,
    horizontal_acceptance_m=DEFAULT_HORIZONTAL_ACCEPTANCE_M,
    vertical_acceptance_m=DEFAULT_VERTICAL_ACCEPTANCE_M,
    run_status=None,
):
    """Derive target-switching evidence from telemetry rows."""
    rows = list(rows)
    modes = {
        _text(row.get("replan_mode")).lower()
        for row in rows
        if _text(row.get("replan_mode"))
    }
    if not modes:
        return _result("UNAVAILABLE", ["replan_mode telemetry is missing"])
    if "active" not in modes:
        return _result("NOT_APPLICABLE", ["run is not active-replan mode"])

    required_columns = ("elapsed_s", "route_direction", "target_name", "phase")
    missing = [name for name in required_columns if not any(name in row for row in rows)]
    transition_index, signal_available = _replacement_index(rows)
    if not signal_available:
        missing.append("replan_route_replaced or active_replan_count")

    if transition_index is None:
        status = "UNAVAILABLE" if missing else "FAIL"
        notes = ["no outbound active route replacement was recorded"]
        if missing:
            notes.append("required telemetry missing: " + ", ".join(missing))
        return _result(status, notes)

    transition_row = rows[transition_index]
    pre_target = ""
    transition_name = _text(transition_row.get("target_name"))
    if WP_PATTERN.fullmatch(transition_name):
        pre_target = transition_name
    else:
        for row in reversed(rows[:transition_index]):
            name = _text(row.get("target_name"))
            if _text(row.get("route_direction")).lower() == "outbound" and name:
                pre_target = name
                break

    post_rows = [
        row
        for row in rows[transition_index + 1 :]
        if _text(row.get("route_direction")).lower() == "outbound"
    ]
    if RWP_PATTERN.fullmatch(transition_name):
        post_rows.insert(0, transition_row)

    collapsed_names = []
    for row in post_rows:
        name = _text(row.get("target_name"))
        if name and (not collapsed_names or name != collapsed_names[-1]):
            collapsed_names.append(name)
    rwp_sequence = [name for name in collapsed_names if RWP_PATTERN.fullmatch(name)]
    rwp_numbers = [int(RWP_PATTERN.fullmatch(name).group(1)) for name in rwp_sequence]
    rwp_contiguous = bool(rwp_numbers) and all(
        current == previous + 1
        for previous, current in zip(rwp_numbers, rwp_numbers[1:])
    )
    old_wp_count = sum(
        1 for row in post_rows if WP_PATTERN.fullmatch(_text(row.get("target_name")))
    )

    first_rwp_name = rwp_sequence[0] if rwp_sequence else ""
    first_rwp_elapsed = None
    if first_rwp_name:
        for row in post_rows:
            if _text(row.get("target_name")) == first_rwp_name:
                first_rwp_elapsed = _number(row.get("elapsed_s"))
                break

    goal_hover_rows = [row for row in rows if _text(row.get("phase")).lower() == "goal_hover"]
    goal_row = goal_hover_rows[0] if goal_hover_rows else None
    final_rwp_rows = [
        row for row in post_rows if RWP_PATTERN.fullmatch(_text(row.get("target_name")))
    ]
    final_rwp_row = final_rwp_rows[-1] if final_rwp_rows else None
    goal_coordinates_available = bool(goal_row and final_rwp_row) and all(
        _number(goal_row.get(key)) is not None
        for key in ("target_north_m", "target_east_m", "target_down_m")
    )
    goal_error_available = bool(goal_hover_rows) and any(
        _number(row.get("horizontal_error_m")) is not None
        and _number(row.get("error_down_m")) is not None
        for row in goal_hover_rows
    )
    original_goal_reached = False
    if goal_coordinates_available and goal_error_available:
        coordinates_match = _same_target_coordinates(final_rwp_row, goal_row)
        accepted = any(
            (_number(row.get("horizontal_error_m")) is not None)
            and (_number(row.get("error_down_m")) is not None)
            and _number(row.get("horizontal_error_m")) < horizontal_acceptance_m
            and abs(_number(row.get("error_down_m"))) < vertical_acceptance_m
            for row in goal_hover_rows
        )
        original_goal_reached = coordinates_match and accepted

    phases = {_text(row.get("phase")).lower() for row in rows}
    failed = bool(phases & {"landing_after_error", "landing_after_danger"})
    runtime_completed = bool(
        run_status
        and run_status.get("status") == "completed"
        and run_status.get("landing_confirmed") is True
    )
    runtime_failed = bool(
        run_status
        and (
            run_status.get("status") != "completed"
            or run_status.get("landing_confirmed") is not True
        )
    )
    mission_completed = ("landed" in phases or runtime_completed) and not failed

    unavailable = list(missing)
    if not pre_target:
        unavailable.append("pre-replan target")
    if not first_rwp_name:
        unavailable.append("first replacement RWP target")
    if not goal_coordinates_available:
        unavailable.append("original-goal target coordinates or final RWP coordinates")
    if not goal_error_available:
        unavailable.append("original-goal horizontal/vertical error")

    violations = []
    if old_wp_count:
        violations.append(
            f"{old_wp_count} old WP telemetry sample(s) reappeared during outbound flight"
        )
    if rwp_sequence and not rwp_contiguous:
        violations.append("distinct outbound RWP sequence is not contiguous")
    if goal_coordinates_available and goal_error_available and not original_goal_reached:
        violations.append("original outbound goal was not reached under the coordinate/error rule")
    if failed:
        violations.append("run contains an error/danger landing phase")
    elif runtime_failed:
        violations.append("runtime status does not confirm a completed landing")
    elif "phase" not in missing and not mission_completed:
        violations.append("run is incomplete because no landed phase was recorded")

    if violations:
        status = "FAIL"
        notes = violations
    elif unavailable:
        status = "UNAVAILABLE"
        notes = ["required evidence missing: " + ", ".join(dict.fromkeys(unavailable))]
    else:
        status = "PASS"
        notes = [
            "outbound target replacement, RWP continuity, original goal, and mission completion verified"
        ]

    return {
        "pre_replan_target_name": pre_target or None,
        "first_replanned_target_name": first_rwp_name or None,
        "first_replanned_target_elapsed_s": first_rwp_elapsed,
        "post_replan_unique_target_sequence": rwp_sequence,
        "post_replan_old_wp_target_count": old_wp_count,
        "rwp_sequence_contiguous": rwp_contiguous if rwp_sequence else None,
        "original_goal_reached": (
            original_goal_reached
            if goal_coordinates_available and goal_error_available
            else None
        ),
        "mission_completed": mission_completed,
        "active_replan_target_switching_status": status,
        "active_replan_target_switching_notes": "; ".join(notes),
    }


def _result(status, notes):
    return {
        "pre_replan_target_name": None,
        "first_replanned_target_name": None,
        "first_replanned_target_elapsed_s": None,
        "post_replan_unique_target_sequence": [],
        "post_replan_old_wp_target_count": 0,
        "rwp_sequence_contiguous": None,
        "original_goal_reached": None,
        "mission_completed": None,
        "active_replan_target_switching_status": status,
        "active_replan_target_switching_notes": "; ".join(notes),
    }
