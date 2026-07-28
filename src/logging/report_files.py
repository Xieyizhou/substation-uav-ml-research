"""CSV, metadata, and manifest writers for A* analysis reports."""

import json

import pandas as pd

from src.logging.log_io import display_path
from src.logging.metrics import (
    duration_s,
    first_active_value,
    first_value,
    perception_enabled_mask,
    safe_max,
)


def save_collision_points_csv(collision_report, output_path):
    rows = collision_report.get("collision_rows", [])
    columns = [
        "entry_type",
        "elapsed_s",
        "local_east_m",
        "local_north_m",
        "grid_x",
        "grid_y",
        "obstacle_names",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"Saved collision points: {output_path}")
    return output_path

def write_run_metadata(
    output_dir,
    run_id,
    stage,
    experiment_type,
    log_path,
    obstacle_config_path,
    df,
    infer_local_replan_enabled_func,
    infer_return_home_enabled_func,
    run_status=None,
):
    metadata = {
        "run_name": run_id,
        "stage": stage,
        "experiment_type": experiment_type,
        "source_log": str(display_path(log_path)),
        "obstacle_config": str(display_path(obstacle_config_path)) if obstacle_config_path else None,
        "perception_enabled": bool(perception_enabled_mask(df).any()) if "perception_enabled" in df.columns else False,
        "risk_action": first_active_value(df, "risk_action") or first_value(df, "risk_action"),
        "local_replan_enabled": infer_local_replan_enabled_func(df),
        "replan_mode": first_value(df, "replan_mode"),
        "altitude_m": safe_max(df, "altitude_m"),
        "max_speed_m_s": safe_max(df, "max_speed_m_s"),
        "return_home": infer_return_home_enabled_func(df),
        "runtime_status": run_status,
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path

def write_manifest(
    output_dir,
    log_path,
    run_id,
    created_at_utc,
    df,
    core_generated_files,
    debug_generated_files,
    collision_generated_files,
    debug_plots_enabled,
    obstacle_config_path,
    warnings,
    obstacle_map,
    collision_report,
    perception_summary_func,
    replan_summary_func,
    active_replan_route_replacement_summary_func,
    active_replan_target_validation_func,
    run_status=None,
):
    manifest_path = output_dir / "manifest.json"
    perception = perception_summary_func(df)
    replan = replan_summary_func(df)
    active_replan_replacement = active_replan_route_replacement_summary_func(df)
    active_replan_validation = active_replan_target_validation_func(df)
    manifest = {
        "run_id": run_id,
        "source_log": str(display_path(log_path)),
        "output_dir": str(display_path(output_dir)),
        "created_at_utc": created_at_utc,
        "run_status": run_status,
        "row_count": len(df),
        "duration_s": duration_s(df),
        "debug_plots_enabled": debug_plots_enabled,
        "planner_name": first_value(df, "planner_name"),
        "map_name": first_value(df, "map_name"),
        "obstacle_config": str(display_path(obstacle_config_path)) if obstacle_config_path else None,
        "height_aware_planning": {
            "altitude_m": obstacle_map.get("altitude_m") if obstacle_map else None,
            "vertical_safety_margin_m": (
                obstacle_map.get("vertical_safety_margin_m") if obstacle_map else None
            ),
            "horizontal_inflation_cells": (
                obstacle_map.get("horizontal_inflation_cells") if obstacle_map else None
            ),
            "blocking_obstacle_names": (
                obstacle_map.get("blocking_obstacle_names") if obstacle_map else []
            ),
            "nonblocking_obstacle_names": (
                obstacle_map.get("nonblocking_obstacle_names") if obstacle_map else []
            ),
            "obstacle_cell_count": obstacle_map.get("obstacle_cell_count") if obstacle_map else None,
            "raw_obstacle_cell_count": (
                obstacle_map.get("raw_obstacle_cell_count") if obstacle_map else None
            ),
            "raw_blocking_cell_count": (
                obstacle_map.get("raw_blocking_cell_count") if obstacle_map else None
            ),
            "inflated_obstacle_cell_count": (
                obstacle_map.get("inflated_obstacle_cell_count") if obstacle_map else None
            ),
        },
        "collision_report": {
            "raw_physical_collision_detected": collision_report["raw_physical_collision_detected"],
            "inflated_safety_buffer_entry_detected": collision_report[
                "inflated_safety_buffer_entry_detected"
            ],
            "raw_collision_points": collision_report["raw_collision_points"],
            "inflated_buffer_entry_points": collision_report["inflated_buffer_entry_points"],
            "first_raw_collision_timestamps": collision_report["first_raw_collision_timestamps"],
            "first_inflated_buffer_entry_timestamps": collision_report[
                "first_inflated_buffer_entry_timestamps"
            ],
            "raw_obstacle_names_involved": collision_report["raw_obstacle_names_involved"],
            "inflated_obstacle_names_involved": collision_report[
                "inflated_obstacle_names_involved"
            ],
        },
        "replan_summary": replan,
        "active_replan_route_replacement_summary": active_replan_replacement,
        "active_replan_target_validation": active_replan_validation,
        "perception_summary": perception,
        "core_generated_files": [str(display_path(path)) for path in core_generated_files],
        "debug_generated_files": [str(display_path(path)) for path in debug_generated_files],
        "collision_generated_files": [
            str(display_path(path)) for path in collision_generated_files
        ],
        "generated_files": [
            str(display_path(path))
            for path in [
                *core_generated_files,
                *debug_generated_files,
                *collision_generated_files,
            ]
        ],
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
