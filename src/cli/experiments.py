"""Official staged experiment commands."""

from __future__ import annotations

import argparse

from src.cli.process import run_shell


STAGES = {
    "static": ("run_static_astar.sh", "Static A* baseline"),
    "perception": ("run_perception_response.sh", "Perception response"),
    "replan-log": ("run_replan_log_only.sh", "Log-only local replanning"),
    "active-replan": ("run_active_replan.sh", "Active local replanning"),
}
EXPERIMENT_DIR = "scripts/flight/experiments"


def print_stages():
    print("\nOfficial Experiment Stages")
    print("=" * 68)
    for stage_id, (_, label) in STAGES.items():
        print(f"  {stage_id:<16} {label}")
    print("=" * 68)
    print("PX4 must already be running before an experiment starts.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py experiment",
        description="Run the four official experiment stages.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="List the official stages")
    run_parser = actions.add_parser("run", help="Run one stage")
    run_parser.add_argument("stage", choices=STAGES)
    all_parser = actions.add_parser("run-all", help="Run every stage repeatedly")
    all_parser.add_argument("--trials", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.action == "list":
        print_stages()
        return 0
    if args.action == "run":
        script, label = STAGES[args.stage]
        return run_shell(
            f"{EXPERIMENT_DIR}/{script}",
            description=f"Running experiment: {label}",
        )
    if args.trials <= 0:
        raise SystemExit("error: --trials must be greater than zero")
    return run_shell(
        f"{EXPERIMENT_DIR}/run_all_3x.sh",
        [str(args.trials)],
        f"Running all experiment stages ({args.trials} trials each)",
    )
