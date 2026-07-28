"""Generate landmark and aggregate cross-stage experiment comparisons."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging.comparison_aggregate import (
    aggregate_csv_value,
    aggregate_md_value,
    aggregate_rows,
    aggregate_stage_row,
    max_value,
    mean_value,
    min_value,
    numeric_values,
    std_value,
    total_value,
    write_aggregate_csv,
    write_aggregate_markdown,
    write_included_runs_csv,
)
from src.logging.comparison_collection import (
    analyzed_run,
    find_all_valid_runs,
    find_marker,
    find_selected_runs,
    infer_experiment_type,
    infer_stage_from_path,
    is_analyzed_run,
    load_json,
    marker_search_roots,
    missing_required_stages,
    print_stage_counts,
    selected_run,
    valid_stage_counts,
)
from src.logging.comparison_landmark import (
    interpretation,
    md_value,
    normalized_value,
    stage_rows,
    write_csv,
    write_markdown,
    write_status,
)
from src.logging.comparison_schema import (
    AGGREGATE_COLUMNS,
    AGGREGATE_COMPARISON_NAME,
    COMPARISON_COLUMNS,
    INCLUDED_RUN_COLUMNS,
    LANDMARK_COMPARISON_NAME,
    MARKERS,
    REQUIRED_STAGES,
    STATUS_FILENAME,
)
from src.logging.evaluation_schema import EVALUATION_COLUMNS
from src.logging.output_registry import (
    RUN_STAGES,
    STAGE_DIR_NAMES,
    STAGE_LABELS,
    display_path,
    ensure_output_tree,
    get_comparisons_dir,
    get_runs_dir,
)
from src.logging.summarize_experiments import (
    collect_run,
    first_non_empty,
    normalized_evaluation_row,
    parse_float,
    read_text,
    to_evaluation_row,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate landmark and aggregate cross-stage comparisons for the "
            "four official experiment stages."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["landmark", "aggregate", "both"],
        default="both",
        help="Comparison output mode. Default: both",
    )
    parser.add_argument(
        "--min-runs-per-stage",
        type=int,
        default=1,
        help="Minimum valid analyzed runs required per official stage. Default: 1",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write comparison outputs even when required stages are missing.",
    )
    parser.add_argument(
        "--strategy",
        choices=["latest-complete"],
        default="latest-complete",
        help="Selection strategy for landmark runs. Default: latest-complete",
    )
    return parser.parse_args()


def write_landmark_outputs(args):
    output_dir = get_comparisons_dir() / LANDMARK_COMPARISON_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = find_selected_runs()
    csv_path = output_dir / "comparison_summary.csv"
    md_path = output_dir / "comparison_summary.md"
    selected_path = output_dir / "selected_runs.json"
    status_path = output_dir / STATUS_FILENAME

    missing_stages = missing_required_stages(rows, 1)
    if missing_stages and not args.allow_partial:
        print("Skipping landmark comparison: required stages are missing selected analyzed runs.")
        print("Missing stages:")
        for stage_name in missing_stages:
            print(f"  - {STAGE_DIR_NAMES[stage_name]}")
        print("Existing landmark comparison_summary.csv and comparison_summary.md were left unchanged.")
        write_status(
            status_path,
            rows,
            missing_stages,
            1,
            generated=False,
        )
        print(f"Wrote {display_path(status_path)}")
        return False

    write_csv(rows, csv_path)
    write_markdown(rows, md_path, csv_path)
    selected_path.write_text(json.dumps(rows, indent=2) + "\n")
    write_status(
        status_path,
        rows,
        missing_stages,
        1,
        generated=True,
    )
    print(f"Selected {len(rows)} landmark run(s).")
    print(f"Wrote {display_path(csv_path)}")
    print(f"Wrote {display_path(md_path)}")
    print(f"Wrote {display_path(selected_path)}")
    print(f"Wrote {display_path(status_path)}")
    return True


def write_aggregate_outputs(all_runs, min_runs_per_stage):
    output_dir = get_comparisons_dir() / AGGREGATE_COMPARISON_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "aggregate_summary.csv"
    md_path = output_dir / "aggregate_summary.md"
    included_runs_path = output_dir / "included_runs.csv"
    missing_stages = missing_required_stages(all_runs, min_runs_per_stage)

    if missing_stages:
        print("Aggregate comparison incomplete: missing required stages or not enough runs.")
        print("Stages below the requested run count:")
        for stage_name in missing_stages:
            print(f"  - {STAGE_DIR_NAMES[stage_name]}")

    summary_rows = aggregate_rows(all_runs)
    write_aggregate_csv(summary_rows, csv_path)
    write_aggregate_markdown(
        summary_rows,
        md_path,
        csv_path,
        included_runs_path,
        missing_stages,
        min_runs_per_stage,
    )
    write_included_runs_csv(all_runs, included_runs_path)
    print(f"Wrote {display_path(csv_path)}")
    print(f"Wrote {display_path(md_path)}")
    print(f"Wrote {display_path(included_runs_path)}")


def main():
    args = parse_args()
    if args.min_runs_per_stage < 1:
        raise ValueError("--min-runs-per-stage must be at least 1")

    ensure_output_tree()
    all_runs = find_all_valid_runs()
    print_stage_counts(all_runs)
    aggregate_missing_stages = missing_required_stages(all_runs, args.min_runs_per_stage)

    success = True
    if args.mode in {"landmark", "both"}:
        if args.mode == "both" and aggregate_missing_stages and not args.allow_partial:
            print(
                "Skipping landmark comparison refresh because the aggregate "
                "stage set is incomplete; existing landmark outputs were left unchanged."
            )
            success = False
        else:
            success = write_landmark_outputs(args) and success
    if args.mode in {"aggregate", "both"}:
        write_aggregate_outputs(all_runs, args.min_runs_per_stage)
        if aggregate_missing_stages and not args.allow_partial:
            success = False
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
