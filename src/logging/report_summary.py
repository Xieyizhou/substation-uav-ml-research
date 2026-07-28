"""Human-readable summary writer for A* analysis reports."""

from src.logging.log_io import display_path
from src.logging.metrics import (
    duration_s,
    first_value,
    format_value,
    safe_last_valid,
    safe_max,
    safe_mean,
)
from src.logging.report_sections import (
    markdown_active_replan_route_replacement_summary,
    markdown_active_replan_target_validation,
    markdown_generated_file_list,
    markdown_perception_summary,
    markdown_replan_summary,
    markdown_transition_table,
)


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
    section_renderers=None,
):
    if section_renderers is None:
        section_renderers = (
            markdown_transition_table,
            markdown_replan_summary,
            markdown_active_replan_route_replacement_summary,
            markdown_active_replan_target_validation,
            markdown_perception_summary,
            markdown_generated_file_list,
        )
    (
        transition_table_renderer,
        replan_renderer,
        active_replan_renderer,
        active_target_renderer,
        perception_renderer,
        generated_files_renderer,
    ) = section_renderers

    summary_path = output_dir / "summary.md"
    planner_name = first_value(df, "planner_name") or "N/A"
    map_name = first_value(df, "map_name") or "N/A"
    transition_rows = waypoint_transition_summary_func(df)
    perception = perception_summary_func(df)
    replan = replan_summary_func(df)
    active_replan_replacement = active_replan_route_replacement_summary_func(df)
    active_replan_validation = active_replan_target_validation_func(df)
    has_obstacle_validation_plot = any(
        path.name == "traj_with_obstacles.png" for path in core_generated_files
    )
    collision_debug_generated = bool(collision_generated_files)

    lines = [
        f"# A* Flight Analysis Summary: {run_id}",
        "",
        f"- Source log: `{display_path(log_path)}`",
        f"- Run ID: `{run_id}`",
        f"- Analysis timestamp: `{created_at_utc}`",
        f"- Runtime status: `{run_status.get('status') if run_status else 'legacy/unavailable'}`",
        f"- Landing confirmed: `{run_status.get('landing_confirmed') if run_status else 'legacy/unavailable'}`",
        f"- Runtime message: `{run_status.get('message') or 'N/A' if run_status else 'N/A'}`",
        f"- Map name: `{map_name}`",
        f"- Planner name: `{planner_name}`",
        f"- Row count: {len(df)}",
        f"- Duration: {format_value(duration_s(df), ' s')}",
        f"- Return-home enabled: `{infer_return_home_enabled_func(df)}`",
        f"- Max altitude: {format_value(safe_max(df, 'altitude_m'), ' m')}",
        f"- Max horizontal error: {format_value(safe_max(df, 'horizontal_error_m'), ' m')}",
        f"- Mean horizontal error: {format_value(safe_mean(df, 'horizontal_error_m'), ' m')}",
        f"- Final horizontal error: {format_value(safe_last_valid(df, 'horizontal_error_m'), ' m')}",
        f"- Waypoint reached threshold: {waypoint_reached_threshold_m:.3f} m",
        f"- Obstacle config: `{display_path(obstacle_config_path) if obstacle_config_path else 'N/A'}`",
        f"- Height-aware altitude: {format_value(obstacle_map.get('altitude_m') if obstacle_map else None, ' m')}",
        f"- Vertical safety margin: {format_value(obstacle_map.get('vertical_safety_margin_m') if obstacle_map else None, ' m')}",
        f"- Horizontal inflation cells: {obstacle_map.get('horizontal_inflation_cells') if obstacle_map else 'N/A'}",
        f"- Blocking obstacles: `{obstacle_map.get('blocking_obstacle_names') if obstacle_map else []}`",
        f"- Ignored/nonblocking obstacles: `{obstacle_map.get('nonblocking_obstacle_names') if obstacle_map else []}`",
        f"- Raw physical footprint cells: {obstacle_map.get('raw_obstacle_cell_count') if obstacle_map else 'N/A'}",
        f"- Height-blocking raw footprint cells: {obstacle_map.get('raw_blocking_cell_count') if obstacle_map else 'N/A'}",
        f"- Inflated blocking cells: {obstacle_map.get('inflated_obstacle_cell_count') if obstacle_map else 'N/A'}",
        f"- raw_physical_collision_detected: {'yes' if collision_report['raw_physical_collision_detected'] else 'no'}",
        f"- inflated_safety_buffer_entry_detected: {'yes' if collision_report['inflated_safety_buffer_entry_detected'] else 'no'}",
        f"- Raw physical footprint entry points: {collision_report['raw_collision_points']}",
        f"- Inflated buffer entry points: {collision_report['inflated_buffer_entry_points']}",
        f"- First raw collision timestamps: {collision_report['first_raw_collision_timestamps']}",
        f"- First inflated buffer entry timestamps: {collision_report['first_inflated_buffer_entry_timestamps']}",
        f"- Raw obstacle names involved: `{collision_report['raw_obstacle_names_involved']}`",
        f"- Inflated obstacle names involved: `{collision_report['inflated_obstacle_names_involved']}`",
        f"- approximate_min_clearance_m: {format_value(collision_report['approximate_min_clearance_m'], ' m')}",
        "",
        "## Collision Interpretation",
        "",
        "- Raw physical footprint entries mean logged local XY entered cells corresponding to Gazebo object footprints.",
        "- Inflated safety-buffer entries mean logged local XY entered cells A* treats as blocked after inflation; this may be a clearance violation rather than a physical collision.",
    ]
    if has_obstacle_validation_plot:
        lines.append(
            "- Coarse grid rectangles can create false positives around thin poles; inspect `traj_with_obstacles.png` and the real SDF geometry."
        )
    else:
        lines.append(
            "- No obstacle validation plot was generated because no obstacle overlay was available."
        )
    if collision_debug_generated:
        lines.append("- Collision debug files were generated because raw physical collision or inflated buffer entries were detected.")
    else:
        lines.append("- No collision/buffer entries detected; collision debug files were not generated.")

    lines.extend([
        "",
        "## Waypoint Transition Summary",
        "",
        *transition_table_renderer(transition_rows),
        "",
        *replan_renderer(replan),
        "",
        *active_replan_renderer(active_replan_replacement),
        "",
        *active_target_renderer(active_replan_validation),
        "",
        *perception_renderer(perception),
        "",
        "## Warnings",
        "",
    ])

    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- None")

    lines.extend(["", "## Generated Files", ""])
    lines.extend(["### Core Files", ""])
    lines.extend(generated_files_renderer(core_generated_files))
    lines.extend(["", "### Debug Files", ""])
    if debug_plots_enabled:
        lines.extend(generated_files_renderer(debug_generated_files))
    else:
        lines.append(
            "- Debug plots were not generated. Re-run with --debug-plots to create altitude, velocity, yaw, waypoint target, and perception count plots."
        )
    lines.extend(["", "### Collision Debug Files", ""])
    lines.extend(generated_files_renderer(collision_generated_files))

    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path
