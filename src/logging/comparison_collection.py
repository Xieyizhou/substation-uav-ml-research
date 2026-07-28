"""Discover and normalize analyzed runs for experiment comparisons."""

import json

from src.logging.comparison_schema import MARKERS, REQUIRED_STAGES
from src.logging.output_registry import (
    RUN_STAGES,
    STAGE_DIR_NAMES,
    display_path,
    get_runs_dir,
)
from src.logging.summarize_experiments import (
    collect_run,
    first_non_empty,
    read_text,
    to_evaluation_row,
)


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        print(f"WARNING: Could not parse {display_path(path)}: {error}")
        return {}


def marker_search_roots():
    return [get_runs_dir(stage_name) for stage_name in RUN_STAGES]


def find_marker(marker_name):
    candidates = []
    for root in marker_search_roots():
        if not root.exists():
            continue
        candidates.extend(root.glob(f"**/{marker_name}"))
    candidates = sorted(
        {path for path in candidates if path.parent.name.startswith("as_")},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def infer_stage_from_path(run_dir, metadata):
    if metadata.get("stage"):
        return metadata["stage"]
    parts = set(run_dir.parts)
    if "01_static_astar" in parts:
        return "static_astar"
    if "02_perception_response" in parts:
        return "perception_response"
    if "03_replan_log_only" in parts:
        return "replan_log_only"
    if "04_active_replan" in parts:
        return "active_replan"
    return "legacy_unmigrated"


def infer_experiment_type(marker_name, metadata):
    if metadata.get("experiment_type"):
        return metadata["experiment_type"]
    mapping = {
        "KEEP_BASELINE__astar_only_perception_disabled.txt": "astar_only_perception_disabled",
        "KEEP_COMPARISON__perception_slow_down.txt": "perception_slow_down",
        "KEEP_CONTROL__log_only_local_replan_no_route_replacement.txt": "local_replan_log_only",
        "KEEP_LANDMARK__active_local_replan.txt": "active_local_replan",
    }
    return mapping.get(marker_name)


def selected_run(marker_name, marker_path):
    run_dir = marker_path.parent
    manifest = load_json(run_dir / "manifest.json")
    metadata = load_json(run_dir / "run_metadata.json")
    summary_text = read_text(run_dir / "summary.md")
    collected = collect_run(run_dir)
    evaluation = to_evaluation_row(collected)
    evaluation.update(
        {
            "stage": infer_stage_from_path(run_dir, metadata),
            "stage_folder": STAGE_DIR_NAMES.get(infer_stage_from_path(run_dir, metadata)),
            "experiment_type": infer_experiment_type(marker_name, metadata),
            "marker": marker_name,
            "run_dir": display_path(run_dir),
            "summary_available": bool(summary_text),
            "manifest_available": bool(manifest),
            "run_metadata_available": bool(metadata),
        }
    )
    if not evaluation.get("run_id"):
        evaluation["run_id"] = first_non_empty(metadata.get("run_name"), collected.get("run_id"), run_dir.name)
    return {
        "marker": marker_name,
        "run_dir": display_path(run_dir),
        "summary_available": bool(summary_text),
        "manifest_available": bool(manifest),
        "run_metadata_available": bool(metadata),
        **evaluation,
    }


def find_selected_runs():
    selected = []
    for marker_name in MARKERS:
        marker_path = find_marker(marker_name)
        if marker_path is None:
            print(f"WARNING: marker not found: {marker_name}")
            continue
        selected.append(selected_run(marker_name, marker_path))
    return selected


def analyzed_run(stage_name, run_dir):
    manifest = load_json(run_dir / "manifest.json")
    metadata = load_json(run_dir / "run_metadata.json")
    summary_text = read_text(run_dir / "summary.md")
    collected = collect_run(run_dir)
    evaluation = to_evaluation_row(collected)
    inferred_stage = evaluation.get("stage") or metadata.get("stage") or stage_name
    evaluation.update(
        {
            "stage": inferred_stage,
            "stage_folder": STAGE_DIR_NAMES.get(inferred_stage),
            "run_dir": display_path(run_dir),
            "summary_available": bool(summary_text),
            "manifest_available": bool(manifest),
            "run_metadata_available": bool(metadata),
        }
    )
    if not evaluation.get("run_id"):
        evaluation["run_id"] = first_non_empty(metadata.get("run_name"), collected.get("run_id"), run_dir.name)
    return evaluation


def find_all_valid_runs():
    rows = []
    for stage_name in RUN_STAGES:
        runs_dir = get_runs_dir(stage_name)
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.glob("as_*")):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "manifest.json").exists() and not (run_dir / "summary.md").exists():
                continue
            row = analyzed_run(stage_name, run_dir)
            if is_analyzed_run(row):
                rows.append(row)
    return rows


def is_analyzed_run(row):
    return bool(row.get("run_id")) and bool(
        row.get("summary_available") or row.get("manifest_available")
    )


def valid_stage_counts(rows):
    counts = {stage_name: 0 for stage_name in REQUIRED_STAGES}
    for row in rows:
        stage_name = row.get("stage")
        if stage_name not in counts:
            continue
        if is_analyzed_run(row):
            counts[stage_name] += 1
    return counts


def missing_required_stages(rows, min_runs_per_stage):
    counts = valid_stage_counts(rows)
    return [
        stage_name
        for stage_name in REQUIRED_STAGES
        if counts.get(stage_name, 0) < min_runs_per_stage
    ]


def print_stage_counts(rows):
    counts = valid_stage_counts(rows)
    for stage_name in REQUIRED_STAGES:
        print(f"{stage_name}: {counts.get(stage_name, 0)} valid run(s)")
