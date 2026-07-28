#!/usr/bin/env python3
"""Validate the latest analyzed active-replan run manifests."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "outputs" / "04_active_replan" / "runs"


def load_eligible_runs(runs_root):
    """Load only analyzed active-mode runs with the new validation payload."""
    eligible = []
    skipped = []
    if not runs_root.exists():
        return eligible, skipped
    for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            skipped.append((run_dir.name, "no manifest.json"))
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            skipped.append((run_dir.name, f"unreadable manifest: {error}"))
            continue
        mode = str((manifest.get("replan_summary") or {}).get("replan_mode") or "").lower()
        validation = manifest.get("active_replan_target_validation")
        if mode != "active":
            skipped.append((run_dir.name, "not active-replan mode"))
            continue
        if not isinstance(validation, dict):
            skipped.append((run_dir.name, "legacy analysis lacks target validation"))
            continue
        eligible.append(
            {
                "run_id": str(manifest.get("run_id") or run_dir.name),
                "validation": validation,
                "manifest_path": manifest_path,
            }
        )
    return eligible, skipped


def evaluate_latest_runs(runs_root, latest=3):
    eligible, skipped = load_eligible_runs(Path(runs_root))
    eligible.sort(key=lambda run: run["run_id"])
    selected = eligible[-latest:]
    all_pass = len(selected) == latest and all(
        run["validation"].get("active_replan_target_switching_status") == "PASS"
        for run in selected
    )
    return selected, eligible, skipped, all_pass


def compact_row(run):
    validation = run["validation"]
    sequence = validation.get("post_replan_unique_target_sequence") or []
    goal = validation.get("original_goal_reached")
    return (
        f"{run['run_id']} | pre={validation.get('pre_replan_target_name') or '-'} "
        f"| first={validation.get('first_replanned_target_name') or '-'} "
        f"| rwp={','.join(sequence) or '-'} "
        f"| old_wp={validation.get('post_replan_old_wp_target_count', '-')} "
        f"| goal={str(goal).lower() if goal is not None else '-'} "
        f"| {validation.get('active_replan_target_switching_status', 'UNAVAILABLE')}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=int, default=3, help="number of latest eligible runs (default: 3)")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.latest <= 0:
        parser.error("--latest must be greater than zero")

    selected, eligible, skipped, passed = evaluate_latest_runs(args.runs_root, args.latest)
    for run in selected:
        print(compact_row(run))
    for run_id, reason in skipped:
        print(f"SKIP {run_id}: {reason}", file=sys.stderr)

    if len(eligible) < args.latest:
        print(f"AGGREGATE FAIL: {len(eligible)}/{args.latest} eligible analyzed active-replan runs")
    elif passed:
        print(f"AGGREGATE PASS: latest {args.latest} eligible runs all passed")
    else:
        print(f"AGGREGATE FAIL: latest {args.latest} eligible runs include FAIL or UNAVAILABLE")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
