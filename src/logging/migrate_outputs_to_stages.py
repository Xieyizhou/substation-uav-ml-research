import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging.output_registry import (
    OUTPUT_ROOT,
    display_path,
    ensure_output_tree,
    get_previews_dir,
    get_run_output_dir,
)


MARKER_STAGE_HINTS = {
    "KEEP_BASELINE__astar_only_perception_disabled.txt": (
        "static_astar",
        "baseline marker for A* with perception disabled",
    ),
    "KEEP_BASELINE__clean_static_astar.txt": (
        "static_astar",
        "legacy baseline marker for clean static A*",
    ),
    "KEEP_CONTROL__perception_log_only_no_behavior_change.txt": (
        "perception_response",
        "control marker for perception log-only without behavior change",
    ),
    "KEEP_COMPARISON__perception_slow_down.txt": (
        "perception_response",
        "comparison marker for perception slow-down",
    ),
    "KEEP_COMPARISON__slow_down_run.txt": (
        "perception_response",
        "legacy marker for perception slow-down",
    ),
    "KEEP_CONTROL__log_only_local_replan_no_route_replacement.txt": (
        "replan_log_only",
        "legacy marker for log-only local replan instrumentation",
    ),
    "KEEP_LANDMARK__active_local_replan.txt": (
        "active_replan",
        "landmark marker for active local replan",
    ),
    "KEEP_LANDMARK__first_active_local_replan.txt": (
        "active_replan",
        "legacy marker for first active local replan",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate legacy top-level outputs into stage folders.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files.")
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        print(f"WARNING: could not parse {display_path(path)}: {error}")
        return {}


def marker_hint(run_dir):
    for marker_name, hint in MARKER_STAGE_HINTS.items():
        if (run_dir / marker_name).exists():
            return hint
    return None, None


def boolish(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def classify_run(run_dir):
    stage, reason = marker_hint(run_dir)
    if stage:
        return stage, reason

    metadata = load_json(run_dir / "run_metadata.json")
    if metadata.get("stage"):
        return metadata["stage"], "existing run_metadata.json stage"

    manifest = load_json(run_dir / "manifest.json")
    perception = manifest.get("perception_summary") or {}
    replan = manifest.get("replan_summary") or {}
    active = manifest.get("active_replan_route_replacement_summary") or {}

    replan_mode = str(replan.get("replan_mode") or active.get("replan_mode") or "").strip()
    total_replans = replan.get("total_replan_attempts") or 0
    active_replacements = (
        active.get("active_route_replacement_count")
        or replan.get("active_route_replacement_count")
        or 0
    )
    try:
        total_replans = int(total_replans)
    except (TypeError, ValueError):
        total_replans = 0
    try:
        active_replacements = int(active_replacements)
    except (TypeError, ValueError):
        active_replacements = 0

    if replan_mode == "active" or active_replacements > 0:
        return "active_replan", "manifest reports active local replanning"
    if total_replans > 0 and replan_mode == "log_only":
        return "replan_log_only", "manifest reports log-only local replan attempts"
    if boolish(perception.get("perception_enabled")):
        return "perception_response", "manifest reports perception enabled without local replan"
    if manifest:
        return "static_astar", "manifest has no enabled perception or local replan evidence"
    return None, "no manifest, metadata, or known marker could classify this run"


def write_stage_note(dest_dir, stage, reason, dry_run):
    note = (
        "# Stage Migration Note\n\n"
        f"- Classified stage: `{stage}`\n"
        f"- Reason: {reason}\n"
        "- Source: legacy top-level `outputs/as_*` folder.\n"
    )
    note_path = dest_dir / "STAGE_NOTE.md"
    if dry_run:
        print(f"DRY-RUN: would write {display_path(note_path)}")
        return
    note_path.write_text(note)


def move_path(source, dest, dry_run):
    if dest.exists():
        print(f"SKIP: destination already exists: {display_path(dest)}")
        return False
    if dry_run:
        print(f"DRY-RUN: would move {display_path(source)} -> {display_path(dest)}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    print(f"Moved {display_path(source)} -> {display_path(dest)}")
    return True


def migrate_preview(source_name, stage_name, dest_name, dry_run):
    source = OUTPUT_ROOT / source_name
    if not source.exists():
        return
    dest = get_previews_dir(stage_name) / dest_name
    move_path(source, dest, dry_run)


def migrate_run(run_dir, dry_run):
    if not run_dir.is_dir() or not run_dir.name.startswith("as_"):
        return
    if run_dir.name == "as_preview":
        return
    stage, reason = classify_run(run_dir)
    if stage is None:
        archive_dest = OUTPUT_ROOT / "archive" / "failed_or_old" / run_dir.name
        print(f"WARNING: uncertain classification for {display_path(run_dir)}: {reason}")
        moved = move_path(run_dir, archive_dest, dry_run)
        if moved:
            write_stage_note(archive_dest, "archive/failed_or_old", reason, dry_run)
        return
    dest = get_run_output_dir(stage, run_dir.name)
    moved = move_path(run_dir, dest, dry_run)
    if moved:
        write_stage_note(dest, stage, reason, dry_run)


def main():
    args = parse_args()
    ensure_output_tree()
    migrate_preview("as_preview", "static_astar", "as_preview", args.dry_run)
    migrate_preview("replan_preview", "replan_log_only", "replan_preview", args.dry_run)

    for run_dir in sorted(OUTPUT_ROOT.glob("as_*")):
        migrate_run(run_dir, args.dry_run)

    if args.dry_run:
        print("Dry run complete; no files were moved.")
    else:
        print("Migration complete.")


if __name__ == "__main__":
    main()
