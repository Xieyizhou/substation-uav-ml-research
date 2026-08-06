"""CLI boundary for frozen multi-scenario visual collection plans."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.vision.collection.plan import (
    collection_status,
    load_collection_plan,
    validate_collection_plan,
    write_collection_plan,
)
from src.vision.collection.recording import (
    collection_recording_context,
    prepare_collection_scenario,
    scenario_by_id,
    validate_collection_recording,
)
from src.vision.collection.dataset import materialize_collection_datasets
from src.vision.collection.batch import run_collection_batch
from src.vision.collection.receipt import (
    write_collection_validation_receipt,
)
from src.vision.collection.pilot_live import (
    append_live_phase_event,
    record_live_visual_pilot,
)
from src.vision.collection.audit import audit_collection_plan
from src.vision.contracts.protocol import protocol_for_plan


DEFAULT_COLLECTION_ROOT = Path("data/research/visual_collection_v1")


def add_collection_parsers(commands):
    plan = commands.add_parser(
        "collection-plan",
        help="Materialize a frozen visual collection plan",
    )
    plan.add_argument(
        "--protocol",
        default="v1",
        help="Registered protocol ID, v1/v2 alias, or protocol JSON path",
    )
    plan.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_COLLECTION_ROOT / "collection_plan.json",
    )
    validate = commands.add_parser(
        "collection-plan-validate",
        help="Validate a frozen visual collection plan",
    )
    validate.add_argument("--input", type=Path, required=True)
    audit = commands.add_parser(
        "collection-audit",
        help="Audit v2 layouts, routes, labels, reachability, and SDF consistency",
    )
    audit.add_argument("--plan", type=Path, required=True)
    status = commands.add_parser(
        "collection-status",
        help=(
            "Show accepted, recording, unvalidated, partial, and missing "
            "recordings"
        ),
    )
    status.add_argument("--plan", type=Path, required=True)
    status.add_argument(
        "--recordings-root",
        type=Path,
        help="Recording directory; defaults beside the selected plan",
    )
    prepare = commands.add_parser(
        "collection-prepare",
        help="Materialize one randomized, reachable visual scenario",
    )
    _add_scenario_arguments(prepare)
    record = commands.add_parser(
        "collection-record",
        help="Record one planned multi-scenario visual run",
    )
    _add_scenario_arguments(record)
    record.add_argument("--scenario-report", type=Path, required=True)
    stopping = record.add_mutually_exclusive_group()
    stopping.add_argument("--duration", type=float)
    stopping.add_argument("--frame-limit", type=int)
    stopping.add_argument("--until-interrupt", action="store_true")
    record.add_argument("--source-timeout", type=float, default=5.0)
    record.add_argument("--max-sync-skew-ms", type=float, default=33.334)
    record.add_argument(
        "--flight-events",
        type=Path,
        help="Flight lifecycle JSONL; defaults inside the recording directory",
    )
    record.add_argument(
        "--manual-lifecycle",
        action="store_true",
        help="Disable automatic phases/landing stop; requires a stopping mode",
    )
    record.add_argument("--post-landing-drain", type=float)
    record.add_argument(
        "--no-auto-validate",
        action="store_false",
        dest="auto_validate",
        default=True,
    )
    phase = commands.add_parser(
        "collection-phase",
        help="Mark one mission phase in the active collection recording",
    )
    _add_scenario_arguments(phase)
    phase.add_argument(
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
    validate_recording = commands.add_parser(
        "collection-record-validate",
        help="Validate one completed visual collection recording",
    )
    _add_scenario_arguments(validate_recording)
    materialize = commands.add_parser(
        "collection-materialize",
        help="Validate all recordings and create split-isolated identities",
    )
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_COLLECTION_ROOT,
    )
    batch = commands.add_parser(
        "collection-run",
        help="Run missing visual collection scenarios sequentially",
    )
    batch.add_argument("--plan", type=Path, required=True)
    batch.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_COLLECTION_ROOT,
    )
    batch.add_argument("--max-scenarios", type=int)
    batch.add_argument("--max-attempts", type=int, default=3)
    batch.add_argument("--retry-delay", type=float, default=10.0)
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--simulator-startup-timeout", type=float, default=180.0)
    batch.add_argument("--probe-timeout", type=float, default=5.0)
    batch.add_argument("--takeoff-ready-timeout", type=float, default=45.0)
    batch.add_argument("--first-frame-timeout", type=float, default=30.0)
    batch.add_argument(
        "--flight-timeout",
        type=float,
        help="Explicit override; v2 otherwise uses its route-derived timeout",
    )
    batch.add_argument("--recorder-timeout", type=float, default=900.0)


def _add_scenario_arguments(parser):
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_COLLECTION_ROOT,
    )


def _scenario_state(args):
    plan = load_collection_plan(args.plan)
    row = scenario_by_id(plan, args.scenario_id)
    recording_directory = (
        args.output_root / "recordings" / row["recording_id"]
    )
    return plan, row, recording_directory


def handle_collection_command(args):
    if args.command == "collection-plan":
        plan = write_collection_plan(args.output, args.protocol)
        return {
            "valid": True,
            "path": str(args.output),
            "protocol_id": plan["protocol_id"],
            "scenario_count": plan["scenario_count"],
            "split_scenario_counts": plan["split_scenario_counts"],
            "collection_plan_identity_sha256": plan[
                "collection_plan_identity_sha256"
            ],
        }
    if args.command == "collection-plan-validate":
        plan = load_collection_plan(args.input)
        return validate_collection_plan(plan)
    if args.command == "collection-audit":
        return audit_collection_plan(load_collection_plan(args.plan))
    if args.command == "collection-status":
        recordings_root = args.recordings_root or args.plan.parent / "recordings"
        return collection_status(
            load_collection_plan(args.plan),
            recordings_root,
        )
    if args.command == "collection-prepare":
        plan, _, _ = _scenario_state(args)
        return prepare_collection_scenario(
            plan,
            args.scenario_id,
            args.output_root,
        )
    if args.command == "collection-record":
        plan, row, recording_directory = _scenario_state(args)
        bounded_stop = any(
            (args.duration is not None, args.frame_limit is not None, args.until_interrupt)
        )
        if args.manual_lifecycle and not bounded_stop:
            raise ValueError("manual collection recording requires a stopping mode")
        if not args.manual_lifecycle and bounded_stop:
            raise ValueError(
                "automatic flight lifecycle cannot use duration, frame limit, "
                "or until-interrupt"
            )
        context = collection_recording_context(
            plan,
            args.scenario_id,
            args.scenario_report,
        )
        flight_events = (
            None
            if args.manual_lifecycle
            else args.flight_events
            or recording_directory / "flight_events.jsonl"
        )
        post_landing_drain = args.post_landing_drain
        if post_landing_drain is None:
            post_landing_drain = protocol_for_plan(plan)["recording"][
                "default_post_landing_drain_s"
            ]
        result = asyncio.run(
            record_live_visual_pilot(
                recording_directory,
                recording_id=row["recording_id"],
                source_timeout_s=args.source_timeout,
                duration_s=args.duration,
                frame_limit=args.frame_limit,
                maximum_skew_ms=args.max_sync_skew_ms,
                recording_context_override=context,
                flight_events_path=flight_events,
                post_landing_drain_s=post_landing_drain,
            )
        )
        lifecycle = result.get("flight_lifecycle") or {}
        if (
            args.auto_validate
            and lifecycle.get("event_type") == "mission_completed"
        ):
            validation = validate_collection_recording(
                recording_directory,
                plan,
            )
            result["validation_receipt"] = (
                write_collection_validation_receipt(
                    recording_directory,
                    plan,
                    validation,
                )
            )
        return result
    if args.command == "collection-phase":
        _, _, recording_directory = _scenario_state(args)
        return {
            "recorded": True,
            "event": append_live_phase_event(
                recording_directory,
                args.phase,
            ),
        }
    if args.command == "collection-record-validate":
        plan, _, recording_directory = _scenario_state(args)
        validation = validate_collection_recording(recording_directory, plan)
        return write_collection_validation_receipt(
            recording_directory,
            plan,
            validation,
        )
    if args.command == "collection-materialize":
        plan = load_collection_plan(args.plan)
        return materialize_collection_datasets(
            plan,
            args.output_root / "recordings",
            args.output_root,
        )
    if args.command == "collection-run":
        plan = load_collection_plan(args.plan)
        return run_collection_batch(
            plan,
            args.plan,
            args.output_root,
            max_scenarios=args.max_scenarios,
            max_attempts=args.max_attempts,
            retry_delay_s=args.retry_delay,
            dry_run=args.dry_run,
            simulator_startup_timeout_s=args.simulator_startup_timeout,
            probe_timeout_s=args.probe_timeout,
            takeoff_ready_timeout_s=args.takeoff_ready_timeout,
            first_frame_timeout_s=args.first_frame_timeout,
            flight_timeout_s=args.flight_timeout,
            recorder_timeout_s=args.recorder_timeout,
        )
    return None
