"""Backward-compatible report writing exports."""

from src.logging.report_files import (
    save_collision_points_csv,
    write_manifest,
    write_run_metadata,
)
from src.logging.report_sections import (
    generated_file_description,
    markdown_active_replan_route_replacement_summary,
    markdown_active_replan_target_validation,
    markdown_generated_file_list,
    markdown_perception_summary,
    markdown_replan_summary,
    markdown_transition_table,
)
from src.logging.report_summary import write_summary as _write_summary


def write_summary(
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
    collision_report,
    warnings,
    obstacle_map,
    waypoint_transition_summary_func,
    perception_summary_func,
    replan_summary_func,
    active_replan_route_replacement_summary_func,
    active_replan_target_validation_func,
    infer_return_home_enabled_func,
    waypoint_reached_threshold_m,
    run_status=None,
):
    """Write summary.md while preserving patchable section renderers."""
    return _write_summary(
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
        collision_report,
        warnings,
        obstacle_map,
        waypoint_transition_summary_func,
        perception_summary_func,
        replan_summary_func,
        active_replan_route_replacement_summary_func,
        active_replan_target_validation_func,
        infer_return_home_enabled_func,
        waypoint_reached_threshold_m,
        run_status,
        section_renderers=(
            markdown_transition_table,
            markdown_replan_summary,
            markdown_active_replan_route_replacement_summary,
            markdown_active_replan_target_validation,
            markdown_perception_summary,
            markdown_generated_file_list,
        ),
    )


__all__ = [
    "generated_file_description",
    "markdown_active_replan_route_replacement_summary",
    "markdown_active_replan_target_validation",
    "markdown_generated_file_list",
    "markdown_perception_summary",
    "markdown_replan_summary",
    "markdown_transition_table",
    "save_collision_points_csv",
    "write_manifest",
    "write_run_metadata",
    "write_summary",
]
