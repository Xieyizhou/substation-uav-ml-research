"""Inspect and validate frozen visual research identities and templates."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.ml.visual_benchmark import (
    VisualBenchmarkResult,
)
from src.ml.visual_benchmark_matrix import (
    load_json,
    validate_static_benchmark_directory,
)
from src.ml.visual_identity import (
    DatasetIdentity,
    ModelIdentity,
    PreprocessingIdentity,
)
from src.ml.visual_pilot_validation import (
    inspect_pilot_recording,
    materialize_pilot_dataset_identity,
    validate_pilot_recording,
)
from src.ml.visual_pilot_live import (
    append_live_phase_event,
    probe_live_visual_sources,
    record_live_visual_pilot,
)
from src.sensors.gazebo_visual_transport import (
    inspect_visual_sources,
    load_research_visual_configuration,
)
from src.cli.visual_collection import (
    add_collection_parsers,
    handle_collection_command,
)
from src.cli.visual_training import (
    add_training_parsers,
    handle_training_command,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "benchmarks/visual_static_v1"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Frozen visual research identity and benchmark tools."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("dataset-validate", "Validate a visual dataset identity"),
        ("dataset-inspect", "Inspect a validated visual dataset identity"),
        ("model-validate", "Validate a visual model identity"),
        ("model-inspect", "Inspect a validated visual model identity"),
        ("preprocessing-validate", "Validate a preprocessing identity"),
        ("preprocessing-inspect", "Inspect a preprocessing identity"),
        ("result-inspect", "Inspect a validated visual benchmark result"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--input", type=Path, required=True)
    matrix = commands.add_parser(
        "matrix-validate", help="Validate the versioned static visual matrix"
    )
    matrix.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    conditions = commands.add_parser(
        "condition-list", help="List unmaterialized static condition templates"
    )
    conditions.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    for name, help_text in (
        ("gazebo-inspect", "Inspect configured live Gazebo visual topics"),
        ("gazebo-probe", "Receive one RGB and truth message with a timeout"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--timeout", type=float, default=5.0)
    pilot_record = commands.add_parser(
        "pilot-record", help="Record one labelled Gazebo visual pilot"
    )
    pilot_record.add_argument("--output", type=Path, required=True)
    pilot_record.add_argument("--recording-id")
    stopping = pilot_record.add_mutually_exclusive_group(required=True)
    stopping.add_argument("--duration", type=float)
    stopping.add_argument("--frame-limit", type=int)
    stopping.add_argument(
        "--until-interrupt",
        action="store_true",
        help="record without a limit until Ctrl-C",
    )
    pilot_record.add_argument("--source-timeout", type=float, default=5.0)
    pilot_record.add_argument("--max-sync-skew-ms", type=float, default=33.334)
    pilot_record.add_argument("--expected-rate-tolerance-fraction", type=float)
    pilot_record.add_argument("--jitter-p95-limit-ms", type=float)
    pilot_record.add_argument("--receive-stall-limit-ms", type=float, default=500.0)
    pilot_record.add_argument("--mission-events", type=Path)
    pilot_phase = commands.add_parser(
        "pilot-phase", help="Mark the current live pilot mission phase"
    )
    pilot_phase.add_argument("--output", type=Path, required=True)
    pilot_phase.add_argument(
        "--phase",
        choices=[
            "cruise_distant",
            "approach",
            "close_inspection",
            "target_transition",
            "other",
        ],
        required=True,
    )
    for name, help_text in (
        ("pilot-inspect", "Inspect a pilot recording, including partial state"),
        ("pilot-validate", "Validate a complete labelled pilot recording"),
        ("pilot-materialize", "Materialize a validated pilot DatasetIdentity"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--input", type=Path, required=True)
    add_collection_parsers(commands)
    add_training_parsers(commands)
    return parser


def _identity_result(command, input_path):
    record = load_json(input_path)
    if command.startswith("dataset-"):
        identity = DatasetIdentity.from_record(record)
        identity_hash = identity.dataset_identity_sha256
        identity_type = "dataset"
    elif command.startswith("model-"):
        identity = ModelIdentity.from_record(record)
        identity_hash = identity.model_identity_sha256
        identity_type = "model"
    else:
        identity = PreprocessingIdentity.from_record(record)
        identity_hash = identity.preprocessing_configuration_id
        identity_type = "preprocessing"
    if command.endswith("-validate"):
        return {
            "valid": True,
            "identity_type": identity_type,
            "identity_sha256": identity_hash,
        }
    return identity.to_record()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command in {
            "dataset-validate",
            "dataset-inspect",
            "model-validate",
            "model-inspect",
            "preprocessing-validate",
            "preprocessing-inspect",
        }:
            result = _identity_result(args.command, args.input)
        elif args.command == "matrix-validate":
            result = validate_static_benchmark_directory(args.benchmark)
            result = {
                key: result[key]
                for key in (
                    "benchmark_id",
                    "materialization_status",
                    "template_count",
                )
            }
            result["valid"] = True
        elif args.command == "condition-list":
            result = validate_static_benchmark_directory(args.benchmark)
            result = {
                "benchmark_id": result["benchmark_id"],
                "materialization_status": result["materialization_status"],
                "templates": result["templates"],
            }
        elif args.command == "result-inspect":
            result = VisualBenchmarkResult.from_record(
                load_json(args.input)
            ).to_record()
        elif args.command == "gazebo-inspect":
            result = {
                "prerequisites": [
                    "Gazebo Sim is already running",
                    "x500_research is spawned",
                    "PX4 may run separately but is not controlled by this command",
                ],
                "configured": load_research_visual_configuration(),
                "live": asyncio.run(
                    inspect_visual_sources(timeout_s=args.timeout)
                ),
            }
        elif args.command == "gazebo-probe":
            result = asyncio.run(
                probe_live_visual_sources(timeout_s=args.timeout)
            )
            result["prerequisites"] = [
                "Gazebo Sim is already running",
                "x500_research is spawned",
            ]
        elif args.command == "pilot-record":
            recording_id = args.recording_id or args.output.name
            print(
                "Prerequisites: Gazebo Sim is already running; x500_research "
                "is spawned; the existing flight task runs separately; this "
                "command does not start PX4, Gazebo, or flight control."
            )
            if args.until_interrupt:
                print(
                    "Recording has no automatic limit; press Ctrl-C once to "
                    "finalize manifests.",
                    flush=True,
                )
            result = asyncio.run(
                record_live_visual_pilot(
                    args.output,
                    recording_id=recording_id,
                    source_timeout_s=args.source_timeout,
                    duration_s=args.duration,
                    frame_limit=args.frame_limit,
                    maximum_skew_ms=args.max_sync_skew_ms,
                    expected_rate_tolerance_fraction=(
                        args.expected_rate_tolerance_fraction
                    ),
                    jitter_p95_limit_ms=args.jitter_p95_limit_ms,
                    receive_stall_limit_ms=args.receive_stall_limit_ms,
                    mission_events_path=args.mission_events,
                )
            )
        elif args.command == "pilot-phase":
            result = {
                "recorded": True,
                "event": append_live_phase_event(args.output, args.phase),
            }
        elif args.command == "pilot-inspect":
            result = inspect_pilot_recording(args.input)
        elif args.command == "pilot-validate":
            result = validate_pilot_recording(args.input)
        elif args.command == "pilot-materialize":
            identity = materialize_pilot_dataset_identity(args.input)
            result = {
                "valid": True,
                "dataset_identity": identity.to_record(),
                "path": str(args.input / "identity/dataset_identity.json"),
            }
        else:
            result = handle_collection_command(args)
            if result is None:
                result = handle_training_command(args)
            if result is None:
                return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        print(f"Visual command failed: {error}")
        return 1
