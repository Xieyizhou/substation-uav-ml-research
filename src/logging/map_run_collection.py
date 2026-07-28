"""Collect cross-map PX4 experiment coverage from analyzed run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.logging.output_registry import OUTPUT_ROOT, RUN_STAGES, display_path, get_runs_dir
from src.logging.summary_collection import collect_run
from src.maps.map_catalog import list_maps, project_path


OUTPUT_DIR = OUTPUT_ROOT / "comparisons" / "maps"
CSV_COLUMNS = [
    "map_id",
    "map_name",
    "difficulty",
    "analyzed_run_count",
    "completed_count",
    "pass_count",
    "stages",
    "latest_run_id",
    "coverage_status",
]


def normalize_map_name(value):
    """Ignore generated map schema versions when matching historical runs."""
    return re.sub(r"_v\d+$", "", str(value or "").strip())


def map_name_index(entries):
    """Map telemetry map names and catalog IDs to canonical catalog IDs."""
    aliases = {}
    for entry in entries:
        aliases[entry["id"]] = entry["id"]
        aliases[normalize_map_name(entry["world_name"])] = entry["id"]
        try:
            config = json.loads(project_path(entry["obstacle_config"]).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = config.get("map_name")
        if name:
            aliases[str(name)] = entry["id"]
            aliases[normalize_map_name(name)] = entry["id"]
    return aliases


def coverage_rows(entries, runs, minimum_runs=1):
    """Return one deterministic coverage row per catalogued map."""
    entries = list(entries)
    aliases = map_name_index(entries)
    grouped = defaultdict(list)
    for run in runs:
        run_map_name = str(run.get("map_name") or "")
        map_id = aliases.get(run_map_name) or aliases.get(normalize_map_name(run_map_name))
        if map_id:
            grouped[map_id].append(run)

    rows = []
    for entry in entries:
        map_runs = grouped[entry["id"]]
        stages = sorted({str(run.get("stage")) for run in map_runs if run.get("stage")})
        latest = max((str(run.get("run_id")) for run in map_runs), default="")
        count = len(map_runs)
        rows.append(
            {
                "map_id": entry["id"],
                "map_name": entry["display_name"],
                "difficulty": entry["difficulty"],
                "analyzed_run_count": count,
                "completed_count": sum(
                    str(run.get("completed_or_failed", "")).lower() == "completed"
                    for run in map_runs
                ),
                "pass_count": sum(
                    str(run.get("final_status", "")).upper() == "PASS"
                    for run in map_runs
                ),
                "stages": ",".join(stages),
                "latest_run_id": latest,
                "coverage_status": "COMPLETE" if count >= minimum_runs else "MISSING",
            }
        )
    return rows


def find_map_runs():
    """Collect runs without dropping map identity during evaluation normalization."""
    rows = []
    for stage in RUN_STAGES:
        runs_dir = get_runs_dir(stage)
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.glob("as_*")):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "manifest.json").exists() and not (
                run_dir / "summary.md"
            ).exists():
                continue
            rows.append(collect_run(run_dir))
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path, csv_path, minimum_runs):
    missing = [row["map_id"] for row in rows if row["coverage_status"] != "COMPLETE"]
    lines = [
        "# Cross-Map PX4 Data Coverage",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "Only analyzed PX4/Gazebo telemetry runs are counted. Offline previews are excluded.",
        f"Required analyzed runs per map: `{minimum_runs}`",
        "",
        "| map | difficulty | runs | completed | pass | stages | latest | coverage |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['map_id']} | {row['difficulty']} | "
            f"{row['analyzed_run_count']} | {row['completed_count']} | "
            f"{row['pass_count']} | {row['stages'] or 'n/a'} | "
            f"{row['latest_run_id'] or 'n/a'} | {row['coverage_status']} |"
        )
    lines.extend(["", f"CSV output: `{display_path(csv_path)}`", ""])
    if missing:
        lines.append("Maps still requiring PX4/Gazebo data: " + ", ".join(f"`{item}`" for item in missing))
    else:
        lines.append("All catalogued maps satisfy the requested coverage threshold.")
    path.write_text("\n".join(lines) + "\n")


def collect_map_coverage(minimum_runs=1, output_dir=OUTPUT_DIR):
    rows = coverage_rows(list_maps(), find_map_runs(), minimum_runs)
    csv_path = Path(output_dir) / "map_run_inventory.csv"
    md_path = Path(output_dir) / "map_run_inventory.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, csv_path, minimum_runs)
    return rows, csv_path, md_path


def build_parser():
    parser = argparse.ArgumentParser(description="Collect analyzed PX4 run coverage by map.")
    parser.add_argument("--min-runs-per-map", type=int, default=1)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when one or more maps are below the requested threshold.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.min_runs_per_map <= 0:
        raise SystemExit("error: --min-runs-per-map must be greater than zero")
    rows, csv_path, md_path = collect_map_coverage(args.min_runs_per_map)
    for row in rows:
        print(
            f"{row['map_id']}: {row['analyzed_run_count']} run(s), "
            f"{row['coverage_status']}"
        )
    print(f"Wrote {display_path(csv_path)}")
    print(f"Wrote {display_path(md_path)}")
    missing = [row for row in rows if row["coverage_status"] != "COMPLETE"]
    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
