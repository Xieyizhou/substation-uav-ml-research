"""Create, resume, compare, and promote versioned ML studies."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from src.ml.artifacts import write_json
from src.ml.model_package import validate_model_package
from src.study.comparison import (
    comparison_report,
    formal_comparison_report,
    study_gate_report,
)
from src.study.registry import ResearchRegistry
from src.study.runner import ingest_results, schedule_tier
from src.study.closed_loop_worker import execute_closed_loop, execute_formal
from src.study.formal_spec import DEFAULT_FORMAL_SPEC


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "outputs/research/registry.sqlite"
DEFAULT_RESULTS = ROOT / "outputs/research/study_results"


def build_parser():
    parser = argparse.ArgumentParser(description="ML research study registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Register a candidate study")
    create.add_argument("--name", required=True)
    create.add_argument("--candidate", type=Path, required=True)
    run = commands.add_parser("run", help="Schedule and ingest one evaluation tier")
    run.add_argument("study_id")
    run.add_argument("--tier", choices=["replay", "closed-loop", "formal"], required=True)
    run.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    execute = commands.add_parser(
        "execute-closed-loop", help="Run pending five-scenario gate flights"
    )
    execute.add_argument("study_id")
    execute.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    execute.add_argument("--max-runs", type=int)
    execute.add_argument("--startup-timeout", type=float, default=180.0)
    execute.add_argument("--probe-timeout", type=float, default=5.0)
    execute.add_argument(
        "--flight-timeout", type=float,
        help="override the default route-aware 240-480 second budget",
    )
    formal = commands.add_parser(
        "execute-formal", help="Run the frozen 120-run paired formal study"
    )
    formal.add_argument("study_id")
    formal.add_argument("--replay-gate", type=Path, required=True)
    formal.add_argument(
        "--qualification-study",
        help="study containing the passed closed-loop candidate gate",
    )
    formal.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    formal.add_argument("--max-runs", type=int)
    formal.add_argument("--startup-timeout", type=float, default=180.0)
    formal.add_argument("--probe-timeout", type=float, default=5.0)
    formal.add_argument("--flight-timeout", type=float)
    formal.add_argument(
        "--comparison-config", type=Path, default=DEFAULT_FORMAL_SPEC
    )
    resume = commands.add_parser("resume", help="Retry incomplete study runs")
    resume.add_argument("study_id")
    resume.add_argument("--tier", choices=["replay", "closed-loop", "formal"])
    resume.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    status = commands.add_parser("status", help="Show study progress")
    status.add_argument("study_id")
    compare = commands.add_parser("compare", help="Build paired metric comparisons")
    compare.add_argument("study_id")
    compare.add_argument("--tier", choices=["replay", "closed-loop", "formal"], default="formal")
    compare.add_argument("--output", type=Path)
    promote = commands.add_parser("promote", help="Promote a safe, improving candidate")
    promote.add_argument("study_id")
    return parser


def _create(registry, args):
    manifest = validate_model_package(args.candidate)
    dataset_manifest = None
    dataset_path = args.candidate
    for path in (ROOT / "data/research").glob("**/dataset_manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset_id") == manifest["dataset_id"]:
            dataset_manifest, dataset_path = value, path.parent
            break
    registry.register_dataset(
        dataset_manifest
        or {
            "dataset_id": manifest["dataset_id"],
            "data_sha256": manifest["dataset_sha256"],
            "source": "model_manifest",
        },
        dataset_path,
    )
    registry.register_model(manifest, args.candidate)
    return {
        "study_id": registry.create_study(
            args.name, manifest["model_id"], manifest.get("parent_model")
        ),
        "candidate_model": manifest["model_id"],
    }


def _status(registry, study_id):
    study = registry.get_study(study_id)
    counts = Counter(
        (run["tier"], run["status"]) for run in registry.runs(study_id)
    )
    study["runs"] = {
        f"{tier}:{status}": count
        for (tier, status), count in sorted(counts.items())
    }
    return study


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        registry = ResearchRegistry(args.registry)
        if args.command == "create":
            result = _create(registry, args)
        elif args.command == "run":
            result = ingest_results(
                registry, args.study_id, args.tier, args.results_dir
            )
        elif args.command == "resume":
            registry.reset_incomplete(args.study_id, args.tier)
            tiers = (args.tier,) if args.tier else ("replay", "closed-loop", "formal")
            result = {
                tier: ingest_results(registry, args.study_id, tier, args.results_dir)
                for tier in tiers
            }
        elif args.command == "execute-closed-loop":
            result = execute_closed_loop(
                args.registry, args.study_id, args.results_dir,
                max_runs=args.max_runs,
                startup_timeout_s=args.startup_timeout,
                probe_timeout_s=args.probe_timeout,
                flight_timeout_s=args.flight_timeout,
            )
        elif args.command == "execute-formal":
            result = execute_formal(
                args.registry, args.study_id, args.results_dir,
                max_runs=args.max_runs,
                startup_timeout_s=args.startup_timeout,
                probe_timeout_s=args.probe_timeout,
                flight_timeout_s=args.flight_timeout,
                replay_gate_path=args.replay_gate,
                comparison_spec_path=args.comparison_config,
                qualification_study_id=args.qualification_study,
            )
        elif args.command == "status":
            result = _status(registry, args.study_id)
        elif args.command == "compare":
            runs = registry.run_metrics(args.study_id, args.tier)
            result = (
                formal_comparison_report(runs)
                if args.tier == "formal"
                else comparison_report(runs)
            )
            if args.output:
                write_json(args.output, result)
        elif args.command == "promote":
            result = study_gate_report(
                {
                    tier: registry.run_metrics(args.study_id, tier)
                    for tier in ("replay", "closed-loop", "formal")
                }
            )
            if result["passed"]:
                registry.promote(args.study_id)
        else:
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if args.command != "promote" or result["passed"] else 1
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Study command failed: {error}")
        return 1
