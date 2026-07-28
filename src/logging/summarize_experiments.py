"""Build per-stage experiment summaries from analyzed run artifacts."""

from src.logging.output_registry import (
    OUTPUT_ROOT,
    RUN_STAGES,
    STAGE_LABELS,
    display_path,
    ensure_output_tree,
    get_summary_paths,
)
from src.logging.summary_values import (
    FINAL_ERROR_PASS_THRESHOLD_M,
    SUMMARY_COLUMNS,
    approximate_distance_traveled,
    choose_status,
    clean_text,
    commanded_speed,
    completed_or_failed,
    count_csv_true,
    count_risk_rows,
    csv_rows,
    first_csv_value,
    first_non_empty,
    format_bool,
    format_float,
    last_csv_float,
    load_json,
    max_csv_float,
    mean_csv_float,
    measured_max_horizontal_speed,
    parse_bool,
    parse_float,
    parse_summary_bullets,
    path_length_from_points,
    planned_path_length,
    read_text,
    resolve_project_path,
    slow_down_event_count,
    target_points,
    warning_note,
    waypoint_reached_count,
)
from src.logging.summary_collection import (
    collect_run,
    find_analysis_dirs,
    find_stage_analysis_dirs,
    infer_stage_from_output_dir,
)
from src.logging.summary_outputs import (
    append_risk_action_comparison,
    average_metric,
    evaluation_markdown_value,
    format_average,
    grouped_by_risk_action,
    markdown_value,
    normalized_csv_row,
    normalized_evaluation_row,
    risk_action_group_stats,
    to_evaluation_row,
    write_csv,
    write_evaluation_csv,
    write_evaluation_markdown,
    write_markdown,
    write_output_index,
)

def main():
    ensure_output_tree()
    stage_counts = {}
    total_runs = 0
    for stage_name in RUN_STAGES:
        analysis_dirs = find_stage_analysis_dirs(stage_name)
        rows = [collect_run(path) for path in analysis_dirs]
        rows.sort(key=lambda row: row.get("run_id") or "")
        paths = get_summary_paths(stage_name)
        write_csv(rows, paths["csv"])
        write_markdown(rows, stage_name, paths["csv"], paths["md"])
        evaluation_rows = [to_evaluation_row(row) for row in rows]
        write_evaluation_csv(evaluation_rows, paths["evaluation_csv"])
        write_evaluation_markdown(
            evaluation_rows,
            stage_name,
            paths["evaluation_csv"],
            paths["evaluation_md"],
        )
        stage_counts[stage_name] = len(rows)
        total_runs += len(rows)
        print(
            f"{stage_name}: found {len(rows)} analyzed A* run(s); "
            f"wrote {display_path(paths['csv'])}, {display_path(paths['md'])}, "
            f"{display_path(paths['evaluation_csv'])}, and {display_path(paths['evaluation_md'])}"
        )

    write_output_index(stage_counts)
    print(f"Found {total_runs} analyzed A* run(s) across stage summaries.")
    print(f"Wrote {display_path(OUTPUT_ROOT / 'README.md')}")


if __name__ == "__main__":
    main()
