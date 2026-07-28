"""Waypoint transition and trajectory warning analysis."""

import math
import re

from src.logging.metrics import bool_from_value, safe_last_valid
from src.logging.plotting import target_sequence

WAYPOINT_REACHED_THRESHOLD_M = 0.4
PLOT_SPLIT_DISTANCE_M = 3.0

def wp_number(target_name):
    match = re.search(r"(\d+)$", str(target_name))
    if not match:
        return None
    return int(match.group(1))


def is_wp_target(target_name):
    return re.match(r"^WP\d+$", str(target_name)) is not None


def is_replanned_wp_target(target_name):
    return re.match(r"^RWP\d+$", str(target_name)) is not None


def active_replan_replacement_events(df):
    required = {"elapsed_s", "target_name", "replan_mode", "replan_route_replaced"}
    if not required.issubset(df.columns):
        return []

    events = []
    previous_name = None
    previous_non_empty_name = None
    previous_route = None
    for _, row in df.dropna(subset=["elapsed_s"]).iterrows():
        name = str(row.get("target_name", "")).strip()
        if name and name != previous_name:
            previous_non_empty_name = previous_name if previous_name else previous_non_empty_name
            previous_name = name
        route = str(row.get("route_direction", "")).strip()
        if route:
            previous_route = route

        replan_mode = str(row.get("replan_mode", "")).strip().lower()
        route_replaced = bool_from_value(row.get("replan_route_replaced"))
        if replan_mode != "active" or not route_replaced:
            continue

        replacement_time = float(row["elapsed_s"])
        target_before = previous_non_empty_name if is_replanned_wp_target(name) else name
        target_before = target_before or previous_non_empty_name
        after_rows = df[
            (df["elapsed_s"] >= replacement_time)
            & (df["target_name"].astype(str).str.strip() != "")
        ]
        replanned_rows = after_rows[
            after_rows["target_name"].astype(str).str.strip().map(is_replanned_wp_target)
        ]
        first_replanned_target = None
        first_replanned_time = None
        if not replanned_rows.empty:
            first_replanned = replanned_rows.iloc[0]
            first_replanned_target = str(first_replanned["target_name"]).strip()
            first_replanned_time = float(first_replanned["elapsed_s"])

        events.append(
            {
                "replacement_time_s": replacement_time,
                "target_before_replacement": target_before,
                "first_replanned_target": first_replanned_target,
                "first_replanned_target_time_s": first_replanned_time,
                "active_replan_path_length": safe_last_valid(row.to_frame().T, "active_replan_path_length"),
                "route_direction": previous_route,
            }
        )
    return events


def expected_active_replan_transition(previous_name, name, elapsed_s, events):
    if not is_wp_target(previous_name) or not is_replanned_wp_target(name):
        return False
    for event in events:
        if event.get("target_before_replacement") != previous_name:
            continue
        if event.get("first_replanned_target") != name:
            continue
        event_time = event.get("replacement_time_s")
        target_time = event.get("first_replanned_target_time_s")
        if event_time is None:
            continue
        if target_time is not None and abs(float(target_time) - float(elapsed_s)) <= 1.0:
            return True
        if 0 <= float(elapsed_s) - float(event_time) <= 2.0:
            return True
    return False


def compute_warnings(df, obstacles, resolution_m, obstacle_config_path, collision_report):
    warnings = []
    position_df = df[["elapsed_s", "local_north_m", "local_east_m"]].dropna()
    jumps = []
    previous = None
    for _, row in position_df.iterrows():
        if previous is not None:
            jump = math.hypot(
                float(row["local_east_m"]) - float(previous["local_east_m"]),
                float(row["local_north_m"]) - float(previous["local_north_m"]),
            )
            if jump > PLOT_SPLIT_DISTANCE_M:
                jumps.append((float(previous["elapsed_s"]), float(row["elapsed_s"]), jump))
        previous = row
    if jumps:
        warnings.append(
            f"{len(jumps)} logged position segment(s) are longer than {PLOT_SPLIT_DISTANCE_M:.1f} m; "
            "plots split these segments to avoid misleading diagonal lines."
        )

    if "target_name" in df.columns:
        active_replan_events = active_replan_replacement_events(df)
        target_changes = df[["elapsed_s", "target_name", "route_direction"]].dropna(subset=["elapsed_s"])
        previous_name = None
        previous_num = None
        previous_route = None
        for _, row in target_changes.iterrows():
            name = str(row["target_name"])
            if not name or name == previous_name:
                continue
            num = wp_number(name)
            route = str(row["route_direction"]) if "route_direction" in target_changes.columns else ""
            # A phase boundary may intentionally restore an original waypoint
            # target (for example RWP06 -> WP09 at goal_hover). Target sequence
            # validation is route-scoped, so do not compare numbering across a
            # route-direction boundary.
            if previous_route is not None and route != previous_route:
                previous_name = name
                previous_num = num
                previous_route = route
                continue
            if previous_num is not None and num is not None:
                expected_delta = -1 if route == "return" or previous_route == "return" else 1
                if num - previous_num not in {expected_delta, 0} and abs(num - previous_num) > 1:
                    if expected_active_replan_transition(
                        previous_name,
                        name,
                        float(row["elapsed_s"]),
                        active_replan_events,
                    ):
                        previous_name = name
                        previous_num = num
                        previous_route = route
                        continue
                    warnings.append(
                        f"Target jumped from {previous_name} to {name} at t={float(row['elapsed_s']):.2f}s; "
                        "check target switching."
                    )
                    break
            previous_name = name
            previous_num = num
            previous_route = route

    first_error = safe_last_valid(df.head(1), "horizontal_error_m")
    targets = target_sequence(df, route_direction="outbound") or target_sequence(df)
    if first_error is not None and len(targets) >= 2:
        full_distance = math.hypot(
            targets[-1]["east_m"] - targets[0]["east_m"],
            targets[-1]["north_m"] - targets[0]["north_m"],
        )
        if full_distance > 0 and first_error > full_distance * 0.75:
            warnings.append(
                "First horizontal error is close to the full start-goal distance; "
                "the initial target may be wrong."
            )

    if collision_report["raw_physical_collision_detected"]:
        warnings.append(
            "Actual trajectory entered raw physical footprint cells; this is more serious than a buffer entry."
        )
    if collision_report["inflated_safety_buffer_entry_detected"]:
        warnings.append(
            "Actual trajectory entered inflated planning obstacle cells; this is a safety-buffer violation."
        )
    elif (
        collision_report["approximate_min_clearance_m"] is not None
        and collision_report["approximate_min_clearance_m"] < 1.0
    ):
        warnings.append(
            "Near-boundary clearance warning: actual trajectory came within about "
            f"{collision_report['approximate_min_clearance_m']:.2f} m of an obstacle cell center."
        )

    return warnings, jumps, collision_report


def waypoint_transition_summary(df):
    required = {"target_name", "elapsed_s", "horizontal_error_m"}
    if not required.issubset(df.columns):
        return []

    rows = []
    grouped = df[df["target_name"].astype(str) != ""].groupby(
        ["route_direction", "target_name"] if "route_direction" in df.columns else ["target_name"],
        dropna=False,
        sort=False,
    )
    for key, group in grouped:
        if isinstance(key, tuple):
            route_direction, target_name = key
        else:
            route_direction, target_name = "", key
        errors = group["horizontal_error_m"].dropna()
        reached_rows = group[group["horizontal_error_m"] <= WAYPOINT_REACHED_THRESHOLD_M]
        rows.append(
            {
                "route_direction": route_direction or "none",
                "target_name": target_name,
                "first_time_s": float(group["elapsed_s"].min()),
                "last_time_s": float(group["elapsed_s"].max()),
                "min_horizontal_error_m": None if errors.empty else float(errors.min()),
                "reached": not reached_rows.empty,
                "reached_time_s": None if reached_rows.empty else float(reached_rows["elapsed_s"].iloc[0]),
            }
        )
    return rows
