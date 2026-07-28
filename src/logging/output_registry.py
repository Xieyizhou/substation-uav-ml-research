from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

STAGE_ROOTS = {
    "static_astar": Path("outputs/01_static_astar"),
    "perception_response": Path("outputs/02_perception_response"),
    "replan_log_only": Path("outputs/03_replan_log_only"),
    "active_replan": Path("outputs/04_active_replan"),
    "archive": Path("outputs/archive"),
    "comparisons": Path("outputs/comparisons"),
}

STAGE_LABELS = {
    "static_astar": "01 static_astar",
    "perception_response": "02 perception_response",
    "replan_log_only": "03 replan_log_only",
    "active_replan": "04 active_replan",
}

STAGE_DIR_NAMES = {
    stage_name: STAGE_ROOTS[stage_name].name
    for stage_name in (
        "static_astar",
        "perception_response",
        "replan_log_only",
        "active_replan",
    )
}

RUN_STAGES = (
    "static_astar",
    "perception_response",
    "replan_log_only",
    "active_replan",
)


def _absolute(relative_path):
    return PROJECT_ROOT / relative_path


def ensure_output_tree():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for stage_name in RUN_STAGES:
        get_runs_dir(stage_name).mkdir(parents=True, exist_ok=True)
        get_previews_dir(stage_name).mkdir(parents=True, exist_ok=True)
        get_summaries_dir(stage_name).mkdir(parents=True, exist_ok=True)
    get_comparisons_dir().mkdir(parents=True, exist_ok=True)
    (_absolute(STAGE_ROOTS["archive"]) / "failed_or_old").mkdir(parents=True, exist_ok=True)


def get_stage_root(stage_name):
    if stage_name not in STAGE_ROOTS:
        raise KeyError(f"Unknown output stage: {stage_name}")
    return _absolute(STAGE_ROOTS[stage_name])


def get_runs_dir(stage_name):
    return get_stage_root(stage_name) / "runs"


def get_previews_dir(stage_name):
    return get_stage_root(stage_name) / "previews"


def get_summaries_dir(stage_name):
    return get_stage_root(stage_name) / "summaries"


def get_run_output_dir(stage_name, run_name):
    return get_runs_dir(stage_name) / run_name


def get_summary_paths(stage_name):
    summaries_dir = get_summaries_dir(stage_name)
    return {
        "csv": summaries_dir / "experiment_summary.csv",
        "md": summaries_dir / "experiment_summary.md",
        "evaluation_csv": summaries_dir / "experiment_evaluation.csv",
        "evaluation_md": summaries_dir / "experiment_evaluation.md",
    }


def get_comparisons_dir():
    return get_stage_root("comparisons")


def display_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except (OSError, ValueError):
        return str(path)
