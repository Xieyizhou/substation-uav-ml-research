"""Perception and replanning summaries for one analyzed run."""

import pandas as pd

from src.logging.analysis_warnings import active_replan_replacement_events
from src.logging.metrics import (
    DEFAULT_DANGER_DISTANCE_M,
    DEFAULT_WARNING_DISTANCE_M,
    RISK_LEVEL_TO_VALUE,
    bool_from_value,
    detected_obstacle_mask,
    duration_s,
    first_value,
    frequent_obstacle_names,
    has_perception_columns,
    perception_enabled_mask,
    ratio,
    risk_level_series,
    safe_last_valid,
    safe_max,
    time_in_risk_levels,
)

WAYPOINT_REACHED_THRESHOLD_M = 0.4

def replan_summary(df):
    if "replan_triggered" not in df.columns:
        return None

    triggered = df["replan_triggered"].map(bool_from_value).fillna(False)
    if "replan_success" in df.columns:
        success = df["replan_success"].map(bool_from_value).fillna(False)
    else:
        success = pd.Series(False, index=df.index)

    total_attempts = int(triggered.sum())
    if "replan_count" in df.columns:
        max_count = df["replan_count"].max(skipna=True)
        if not pd.isna(max_count):
            total_attempts = max(total_attempts, int(max_count))

    route_replaced = pd.Series(False, index=df.index)
    active_replacement_count = None
    max_active_path_length = None
    replan_mode = first_value(df, "replan_mode") or "log_only"
    if "replan_route_replaced" in df.columns:
        route_replaced = df["replan_route_replaced"].map(bool_from_value).fillna(False)
        active_replacement_count = int(route_replaced.sum())
    if "active_replan_count" in df.columns:
        max_active_count = safe_max(df, "active_replan_count")
        if max_active_count is not None:
            active_replacement_count = max(active_replacement_count or 0, int(max_active_count))
    if "active_replan_path_length" in df.columns and route_replaced.any():
        max_active_path_length = safe_max(df[route_replaced], "active_replan_path_length")

    return {
        "available": True,
        "replan_mode": replan_mode,
        "total_replan_attempts": total_attempts,
        "successful_replan_attempts": int((triggered & success).sum()),
        "max_replan_path_length": safe_max(df[triggered], "replan_path_length") if triggered.any() else None,
        "max_dynamic_blocked_cell_count": (
            safe_max(df[triggered], "dynamic_blocked_cell_count") if triggered.any() else None
        ),
        "active_route_replacement_count": active_replacement_count,
        "max_active_replan_path_length": max_active_path_length,
    }


def active_replan_route_replacement_summary(df):
    if "replan_mode" not in df.columns and "replan_route_replaced" not in df.columns:
        return None

    events = active_replan_replacement_events(df)
    replan_mode = first_value(df, "replan_mode") or "log_only"
    route_replaced = pd.Series(False, index=df.index)
    if "replan_route_replaced" in df.columns:
        route_replaced = df["replan_route_replaced"].map(bool_from_value).fillna(False)

    replacement_count = int(route_replaced.sum())
    if "active_replan_count" in df.columns:
        max_active_count = safe_max(df, "active_replan_count")
        if max_active_count is not None:
            replacement_count = max(replacement_count, int(max_active_count))
    replacement_count = max(replacement_count, len(events))

    first_event = events[0] if events else {}
    active_path_length = first_event.get("active_replan_path_length")
    if active_path_length is None and "active_replan_path_length" in df.columns and route_replaced.any():
        active_path_length = safe_last_valid(df[route_replaced].head(1), "active_replan_path_length")
    if active_path_length is None and "active_replan_path_length" in df.columns:
        active_path_length = safe_max(df, "active_replan_path_length")

    return {
        "available": True,
        "replan_mode": replan_mode,
        "active_route_replacement_count": replacement_count,
        "first_active_route_replacement_time_s": first_event.get("replacement_time_s"),
        "target_before_replacement": first_event.get("target_before_replacement"),
        "first_replanned_target_after_replacement": first_event.get("first_replanned_target"),
        "active_replan_path_length": active_path_length,
        "max_active_replan_path_length": safe_max(df, "active_replan_path_length")
        if "active_replan_path_length" in df.columns
        else None,
    }


def perception_summary(df):
    summary = {
        "available": has_perception_columns(df),
        "perception_enabled": False,
        "total_duration_s": duration_s(df),
        "detection_range_m": None,
        "detection_fov_deg": None,
        "warning_distance_m": None,
        "danger_distance_m": None,
        "risk_action": None,
        "perception_source": None,
        "sensor_healthy_ratio": None,
        "sensor_frequency_p50_hz": None,
        "sensor_frequency_p95_hz": None,
        "sensor_frame_age_p50_ms": None,
        "sensor_frame_age_p95_ms": None,
        "sensor_dropped_frames_max": None,
        "risk_model_id": None,
        "inference_latency_p50_ms": None,
        "inference_latency_p95_ms": None,
        "costmap_version_max": None,
        "equipment_model_id": None,
        "equipment_detection_count_max": None,
        "equipment_classes": [],
        "total_detection_samples": 0,
        "samples_with_detections": 0,
        "clear_sample_count": 0,
        "detected_sample_count": 0,
        "warning_sample_count": 0,
        "danger_sample_count": 0,
        "clear_sample_ratio": None,
        "detected_sample_ratio": None,
        "warning_sample_ratio": None,
        "danger_sample_ratio": None,
        "time_in_clear_s": 0.0,
        "time_in_detected_s": 0.0,
        "time_in_warning_s": 0.0,
        "time_in_danger_s": 0.0,
        "percent_time_clear": None,
        "percent_time_detected": None,
        "percent_time_warning": None,
        "percent_time_danger": None,
        "clear_samples": 0,
        "detected_samples": 0,
        "warning_samples": 0,
        "danger_samples": 0,
        "first_detection_time_s": None,
        "first_warning_time_s": None,
        "first_danger_time_s": None,
        "nearest_obstacle_ever_detected": None,
        "minimum_nearest_obstacle_distance_m": None,
        "mean_nearest_obstacle_distance_m": None,
        "median_nearest_obstacle_distance_m": None,
        "most_frequent_obstacle_names": [],
        "most_frequent_warning_obstacle_names": [],
        "most_frequent_danger_obstacle_names": [],
    }
    if not summary["available"]:
        return summary

    enabled_mask = perception_enabled_mask(df)
    detected_mask = detected_obstacle_mask(df)
    active_df = df[enabled_mask]
    detected_df = df[enabled_mask & detected_mask]

    summary["perception_enabled"] = bool(enabled_mask.any())
    summary["total_detection_samples"] = int(len(active_df))
    summary["samples_with_detections"] = int(len(detected_df))
    detection_range = first_value(active_df, "detection_range_m")
    detection_fov = first_value(active_df, "detection_fov_deg")
    summary["detection_range_m"] = None if detection_range is None else float(detection_range)
    summary["detection_fov_deg"] = None if detection_fov is None else float(detection_fov)
    warning_distance = first_value(active_df, "warning_distance_m")
    danger_distance = first_value(active_df, "danger_distance_m")
    summary["warning_distance_m"] = (
        DEFAULT_WARNING_DISTANCE_M if warning_distance is None else float(warning_distance)
    )
    summary["danger_distance_m"] = (
        DEFAULT_DANGER_DISTANCE_M if danger_distance is None else float(danger_distance)
    )
    summary["risk_action"] = first_value(active_df, "risk_action") or "log_only"
    summary["perception_source"] = first_value(active_df, "perception_source")
    summary["risk_model_id"] = first_value(active_df, "risk_model_id")
    summary["equipment_model_id"] = first_value(active_df, "equipment_model_id")

    if "sensor_healthy" in active_df.columns:
        health = active_df["sensor_healthy"].map(bool_from_value).dropna()
        summary["sensor_healthy_ratio"] = float(health.mean()) if not health.empty else None

    def numeric_quantiles(column, *, scale=1.0):
        if column not in active_df.columns:
            return None, None
        values = pd.to_numeric(active_df[column], errors="coerce").dropna() * scale
        if values.empty:
            return None, None
        return float(values.quantile(0.5)), float(values.quantile(0.95))

    (
        summary["sensor_frequency_p50_hz"],
        summary["sensor_frequency_p95_hz"],
    ) = numeric_quantiles("sensor_frequency_hz")
    (
        summary["sensor_frame_age_p50_ms"],
        summary["sensor_frame_age_p95_ms"],
    ) = numeric_quantiles("sensor_frame_age_s", scale=1000.0)
    (
        summary["inference_latency_p50_ms"],
        summary["inference_latency_p95_ms"],
    ) = numeric_quantiles("inference_latency_ms")
    summary["sensor_dropped_frames_max"] = safe_max(
        active_df, "sensor_dropped_frames"
    )
    summary["costmap_version_max"] = safe_max(active_df, "costmap_version")
    summary["equipment_detection_count_max"] = safe_max(
        active_df, "equipment_detection_count"
    )
    if "equipment_classes" in active_df.columns:
        summary["equipment_classes"] = sorted(
            {
                item.strip()
                for value in active_df["equipment_classes"].dropna().astype(str)
                for item in value.split("|")
                if item.strip()
            }
        )

    risk_levels = risk_level_series(active_df)
    summary["clear_sample_count"] = int((risk_levels == "clear").sum())
    summary["detected_sample_count"] = int((risk_levels == "detected").sum())
    summary["warning_sample_count"] = int((risk_levels == "warning").sum())
    summary["danger_sample_count"] = int((risk_levels == "danger").sum())
    summary["clear_samples"] = summary["clear_sample_count"]
    summary["detected_samples"] = summary["detected_sample_count"]
    summary["warning_samples"] = summary["warning_sample_count"]
    summary["danger_samples"] = summary["danger_sample_count"]
    total_samples = summary["total_detection_samples"]
    summary["clear_sample_ratio"] = ratio(summary["clear_sample_count"], total_samples)
    summary["detected_sample_ratio"] = ratio(summary["detected_sample_count"], total_samples)
    summary["warning_sample_ratio"] = ratio(summary["warning_sample_count"], total_samples)
    summary["danger_sample_ratio"] = ratio(summary["danger_sample_count"], total_samples)

    risk_time = time_in_risk_levels(active_df, risk_levels)
    for level in RISK_LEVEL_TO_VALUE:
        summary[f"time_in_{level}_s"] = risk_time[level]
    risk_time_total = sum(risk_time.values())
    for level in RISK_LEVEL_TO_VALUE:
        value = ratio(risk_time[level], risk_time_total)
        summary[f"percent_time_{level}"] = None if value is None else value * 100.0

    warning_df = active_df[risk_levels == "warning"]
    danger_df = active_df[risk_levels == "danger"]
    if not warning_df.empty:
        summary["first_warning_time_s"] = safe_last_valid(warning_df.head(1), "elapsed_s")
    if not danger_df.empty:
        summary["first_danger_time_s"] = safe_last_valid(danger_df.head(1), "elapsed_s")

    if detected_df.empty:
        return summary

    summary["first_detection_time_s"] = safe_last_valid(detected_df.head(1), "elapsed_s")
    distances = detected_df["nearest_obstacle_distance_m"].dropna()
    if not distances.empty:
        min_index = distances.idxmin()
        summary["minimum_nearest_obstacle_distance_m"] = float(distances.loc[min_index])
        summary["mean_nearest_obstacle_distance_m"] = float(distances.mean())
        summary["median_nearest_obstacle_distance_m"] = float(distances.median())
        summary["nearest_obstacle_ever_detected"] = str(
            detected_df.loc[min_index].get("nearest_obstacle_name", "")
        )

    summary["most_frequent_obstacle_names"] = frequent_obstacle_names(detected_df)
    summary["most_frequent_warning_obstacle_names"] = frequent_obstacle_names(warning_df)
    summary["most_frequent_danger_obstacle_names"] = frequent_obstacle_names(danger_df)
    return summary
