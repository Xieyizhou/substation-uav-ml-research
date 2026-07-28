import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.obstacle_config import build_obstacle_map, get_resolution_altitude, load_obstacle_config
from src.logging.collision_checks import obstacle_collision_report
from src.logging.active_replan_validation import validate_active_replan_rows
from src.logging.log_io import (
    display_path,
    prepare_dataframe,
    resolve_log_path,
    resolve_project_path,
    run_id_from_log,
)
from src.logging.metrics import (
    DEFAULT_DANGER_DISTANCE_M,
    DEFAULT_WARNING_DISTANCE_M,
    RISK_LEVEL_TO_VALUE,
    bool_from_value,
    detected_obstacle_mask,
    duration_s,
    first_value,
    first_active_value,
    frequent_obstacle_names,
    has_perception_columns,
    perception_enabled_mask,
    ratio,
    risk_level_series,
    safe_last_valid,
    safe_max,
    safe_median,
    time_in_risk_levels,
)
from src.logging.output_registry import ensure_output_tree, get_run_output_dir


OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MAX_ASTAR_OUTPUT_FOLDERS = 10
ASTAR_OUTPUT_PATTERN = re.compile(r"^as_\d{8}_\d{6}$")
WAYPOINT_REACHED_THRESHOLD_M = 0.4
PLOT_SPLIT_DISTANCE_M = 3.0
# Keep matplotlib cache files inside this project instead of relying on a
# writable home-directory cache.
MPLCONFIG_DIR = OUTPUT_ROOT / ".matplotlib_cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

from src.logging.plotting import (
    save_collision_zoom_plot,
    save_detection_count_plot,
    save_error_plot,
    save_line_plot,
    save_perception_risk_timeline,
    save_perception_timeline,
    save_target_timeline,
    save_trajectory_plot,
    target_sequence,
)
from src.logging.report_writer import (
    save_collision_points_csv,
    write_manifest,
    write_run_metadata,
    write_summary,
)



from src.logging.analysis_inference import (
    infer_experiment_stage,
    infer_experiment_type,
    infer_local_replan_enabled,
    infer_return_home_enabled,
    load_analysis_obstacles,
    obstacle_cell_names,
)
from src.logging.analysis_warnings import (
    active_replan_replacement_events,
    compute_warnings,
    expected_active_replan_transition,
    is_replanned_wp_target,
    is_wp_target,
    waypoint_transition_summary,
    wp_number,
)
from src.logging.analysis_summaries import (
    active_replan_route_replacement_summary,
    perception_summary,
    replan_summary,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze an A* path flight log.")
    parser.add_argument(
        "--log",
        type=Path,
        help="Path to an astar_*.csv log. If omitted, the newest log is used.",
    )
    parser.add_argument(
        "--obstacle-config",
        type=Path,
        help="Optional JSON obstacle config, for example config/substation_obstacles.json.",
    )
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        help=(
            "Also generate altitude, velocity, yaw, waypoint target, nearest-obstacle, "
            "and detection-count debug plots. Default analysis only writes core plots."
        ),
    )
    return parser.parse_args()


def cleanup_old_astar_outputs():
    return []


def load_run_status(log_path, warnings):
    status_path = log_path.with_suffix(".status.json")
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"Could not read run status {display_path(status_path)}: {error}")
        return None


def main():
    args = parse_args()
    ensure_output_tree()
    try:
        log_path, used_newest_log = resolve_log_path(args.log)
    except FileNotFoundError as error:
        print(f"Analysis skipped: {error}")
        print("Run an A* flight first, or pass --log path/to/astar_*.csv.")
        return 2

    run_id = run_id_from_log(log_path)
    df = prepare_dataframe(log_path)
    stage = infer_experiment_stage(df)
    experiment_type = infer_experiment_type(df, stage)
    output_dir = get_run_output_dir(stage, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at_utc = datetime.now(timezone.utc).isoformat()
    initial_warnings = []
    run_status = load_run_status(log_path, initial_warnings)
    if run_status and (
        run_status.get("status") != "completed"
        or run_status.get("landing_confirmed") is not True
    ):
        initial_warnings.append(
            "Flight runtime status is not a confirmed completion: "
            f"status={run_status.get('status')}, "
            f"landing_confirmed={run_status.get('landing_confirmed')}, "
            f"message={run_status.get('message') or 'N/A'}"
        )
    obstacle_config_path, obstacle_config, obstacles, obstacle_resolution_m, obstacle_map = load_analysis_obstacles(
        args, df, initial_warnings
    )
    if obstacle_config and (first_value(df, "map_name") is None):
        df["map_name"] = obstacle_config.get("map_name", "")
    log_resolution = safe_last_valid(df, "resolution_m")
    resolution_m = obstacle_resolution_m or log_resolution or 1.0
    collision_report = obstacle_collision_report(
        df,
        obstacle_map,
        resolution_m,
    )
    warnings, jumps, collision_report = compute_warnings(
        df,
        obstacles,
        resolution_m,
        obstacle_config_path,
        collision_report,
    )
    warnings = [*initial_warnings, *warnings]

    print(f"Analyzing log: {display_path(log_path)}")
    print(f"Run ID: {run_id}")
    print(f"Output stage: {stage}")
    print(f"Rows: {len(df)}")
    if obstacle_map:
        print("Height-aware analysis:")
        print(f"  flight altitude: {obstacle_map['altitude_m']} m")
        print(f"  vertical safety margin: {obstacle_map['vertical_safety_margin_m']} m")
        print(f"  horizontal inflation cells: {obstacle_map['horizontal_inflation_cells']}")
        print(f"  blocking obstacles: {obstacle_map['blocking_obstacle_names']}")
        print(f"  ignored/nonblocking obstacles: {obstacle_map['nonblocking_obstacle_names']}")
    if warnings:
        print("Analysis warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    core_generated_files = []
    debug_generated_files = []
    collision_generated_files = []
    source_log = str(display_path(log_path))

    if args.debug_plots:
        plot_path = save_line_plot(
            df,
            ["altitude_m"],
            "Altitude Over Time",
            "Altitude above local origin (m)",
            output_dir / "alt.png",
            source_log,
        )
        if plot_path is not None:
            debug_generated_files.append(plot_path)

    map_name = first_value(df, "map_name")
    plot_path = save_trajectory_plot(
        df,
        output_dir / "traj.png",
        log_path,
        map_name,
        obstacle_map,
        resolution_m,
        collision_report,
        title="A* Local 2D Trajectory",
        show_obstacles=False,
        show_collision_points=False,
    )
    if plot_path is not None:
        core_generated_files.append(plot_path)

    if obstacle_config_path is not None and obstacle_map:
        plot_path = save_trajectory_plot(
            df,
            output_dir / "traj_with_obstacles.png",
            log_path,
            map_name,
            obstacle_map,
            resolution_m,
            collision_report,
            title="A* Local 2D Trajectory with Obstacle Validation",
            show_obstacles=True,
            show_collision_points=True,
            show_perception_points=True,
        )
        if plot_path is not None:
            core_generated_files.append(plot_path)
    else:
        print(
            "Skipping traj_with_obstacles.png: pass --obstacle-config to enable the obstacle validation overlay."
        )

    plot_path = save_error_plot(df, output_dir / "error.png", source_log)
    if plot_path is not None:
        core_generated_files.append(plot_path)

    if args.debug_plots:
        plot_path = save_line_plot(
            df,
            ["velocity_north_m_s", "velocity_east_m_s", "velocity_down_m_s"],
            "NED Velocity Over Time",
            "Velocity (m/s)",
            output_dir / "vel.png",
            source_log,
        )
        if plot_path is not None:
            debug_generated_files.append(plot_path)

        plot_path = save_line_plot(
            df,
            ["yaw_deg"],
            "Yaw Over Time",
            "Yaw (deg)",
            output_dir / "yaw.png",
            source_log,
        )
        if plot_path is not None:
            debug_generated_files.append(plot_path)

        plot_path = save_target_timeline(df, output_dir / "target_timeline.png", source_log)
        if plot_path is not None:
            debug_generated_files.append(plot_path)

    if has_perception_columns(df):
        plot_path = save_perception_risk_timeline(
            df,
            output_dir / "perception_risk_timeline.png",
            source_log,
        )
        if plot_path is not None:
            core_generated_files.append(plot_path)

        if args.debug_plots:
            plot_path = save_perception_timeline(
                df,
                output_dir / "perception_timeline.png",
                source_log,
            )
            if plot_path is not None:
                debug_generated_files.append(plot_path)

            plot_path = save_detection_count_plot(
                df,
                output_dir / "detection_count_over_time.png",
                source_log,
            )
            if plot_path is not None:
                debug_generated_files.append(plot_path)

    if collision_report["obstacle_collision_detected"]:
        collision_csv = save_collision_points_csv(
            collision_report,
            output_dir / "collision_points.csv",
        )
        if collision_csv is not None:
            collision_generated_files.append(collision_csv)

        collision_zoom = save_collision_zoom_plot(
            df,
            output_dir / "collision_zoom.png",
            obstacle_map,
            resolution_m,
            collision_report,
        )
        if collision_zoom is not None:
            collision_generated_files.append(collision_zoom)
    else:
        print("No collision/buffer entries detected; collision debug files were not generated.")

    summary_path = output_dir / "summary.md"
    manifest_path = output_dir / "manifest.json"
    metadata_path = output_dir / "run_metadata.json"
    core_generated_files = [summary_path, manifest_path, metadata_path, *core_generated_files]
    summary_path = write_summary(
        output_dir=output_dir,
        log_path=log_path,
        run_id=run_id,
        created_at_utc=created_at_utc,
        df=df,
        core_generated_files=core_generated_files,
        debug_generated_files=debug_generated_files,
        collision_generated_files=collision_generated_files,
        debug_plots_enabled=args.debug_plots,
        obstacle_config_path=obstacle_config_path,
        collision_report=collision_report,
        warnings=warnings,
        obstacle_map=obstacle_map,
        waypoint_transition_summary_func=waypoint_transition_summary,
        perception_summary_func=perception_summary,
        replan_summary_func=replan_summary,
        active_replan_route_replacement_summary_func=active_replan_route_replacement_summary,
        active_replan_target_validation_func=lambda frame: validate_active_replan_rows(
            frame.to_dict(orient="records"),
            run_status=run_status,
        ),
        infer_return_home_enabled_func=infer_return_home_enabled,
        waypoint_reached_threshold_m=WAYPOINT_REACHED_THRESHOLD_M,
        run_status=run_status,
    )
    manifest_path = write_manifest(
        output_dir=output_dir,
        log_path=log_path,
        run_id=run_id,
        created_at_utc=created_at_utc,
        df=df,
        core_generated_files=core_generated_files,
        debug_generated_files=debug_generated_files,
        collision_generated_files=collision_generated_files,
        debug_plots_enabled=args.debug_plots,
        obstacle_config_path=obstacle_config_path,
        warnings=warnings,
        obstacle_map=obstacle_map,
        collision_report=collision_report,
        perception_summary_func=perception_summary,
        replan_summary_func=replan_summary,
        active_replan_route_replacement_summary_func=active_replan_route_replacement_summary,
        active_replan_target_validation_func=lambda frame: validate_active_replan_rows(
            frame.to_dict(orient="records"),
            run_status=run_status,
        ),
        run_status=run_status,
    )
    metadata_path = write_run_metadata(
        output_dir=output_dir,
        run_id=run_id,
        stage=stage,
        experiment_type=experiment_type,
        log_path=log_path,
        obstacle_config_path=obstacle_config_path,
        df=df,
        infer_local_replan_enabled_func=infer_local_replan_enabled,
        infer_return_home_enabled_func=infer_return_home_enabled,
        run_status=run_status,
    )
    generated_files = [
        *core_generated_files,
        *debug_generated_files,
        *collision_generated_files,
    ]

    output_dir.touch()
    cleanup_old_astar_outputs()

    print("\nAnalysis complete.")
    print(f"Source log: {display_path(log_path)}")
    print(f"Output folder: {display_path(output_dir)}")
    print("Generated files:")
    for path in generated_files:
        print(f"* {display_path(path)}")

    if used_newest_log:
        print("\nTo analyze this exact run again:")
        command = f"python main.py report analyze --log {display_path(log_path)}"
        if obstacle_config_path:
            command += f" --obstacle-config {display_path(obstacle_config_path)}"
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
