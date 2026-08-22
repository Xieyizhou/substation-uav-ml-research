"""CLI boundary for licensed real-domain visual data and YOLO11n v3."""

from __future__ import annotations

from pathlib import Path

from src.vision.evaluation.real_blind_view import materialize_real_blind_view
from src.vision.evaluation.v3_gate import (
    evaluate_synthetic_regression,
    validate_v3_candidate,
)
from src.vision.training.real_domain_dataset import (
    audit_real_dataset,
    inspect_real_dataset,
    materialize_real_dataset,
)
from src.vision.training.source_download import download_approved_source
from src.vision.training.source_export_audit import audit_source_export
from src.vision.training.source_content_audit import audit_source_content
from src.vision.training.source_feasibility import audit_real_sources
from src.vision.training.source_intake import intake_real_source
from src.vision.training.gomes_yolo_audit import audit_gomes_yolo_component
from src.vision.training.gomes_family_audit import audit_gomes_source_family
from src.vision.training.v3_view import materialize_real_training_view


DEFAULT_REGISTRY = Path("config/perception/real_domain_sources.json")
DEFAULT_MANIFEST = Path("data/external/real_domain_v3/samples.jsonl")
DEFAULT_SOURCE = Path("data/external/real_domain_v3/source")
DEFAULT_DATASET = Path("data/research/real_domain_v3")
DEFAULT_V3_VIEW = Path("data/research/visual_yolo_v3")
DEFAULT_AUDIT = Path("outputs/research/real_domain_source_audit")
DEFAULT_INTAKE = Path("outputs/research/real_domain_quarantine")
DEFAULT_DOWNLOADS = Path("data/external/real_domain_v3/downloads")


def add_real_domain_parsers(commands):
    source_audit = commands.add_parser(
        "real-source-audit", help="Evaluate public sources before downloading"
    )
    source_audit.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    source_audit.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    source_audit.add_argument("--content-audits", type=Path, default=DEFAULT_INTAKE)

    intake = commands.add_parser(
        "real-source-intake", help="Fetch bounded public previews into quarantine"
    )
    intake.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    intake.add_argument("--source-id", required=True)
    intake.add_argument("--output", type=Path, default=DEFAULT_INTAKE)
    intake.add_argument("--sample-limit", type=int, default=8)

    export_audit = commands.add_parser(
        "real-source-export-audit", help="Inspect a manually exported YOLO ZIP"
    )
    export_audit.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    export_audit.add_argument("--source-id", required=True)
    export_audit.add_argument("--archive", type=Path, required=True)
    export_audit.add_argument("--output", type=Path, default=DEFAULT_INTAKE)

    content_audit = commands.add_parser(
        "real-source-content-audit", help="Audit uniqueness and semantics in a YOLO ZIP"
    )
    content_audit.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    content_audit.add_argument("--source-id", required=True)
    content_audit.add_argument("--archive", type=Path, required=True)
    content_audit.add_argument("--output", type=Path, default=DEFAULT_INTAKE)

    gomes_audit = commands.add_parser(
        "real-gomes-yolo-audit", help="Audit an official Gomes Figshare capture group"
    )
    gomes_audit.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    gomes_audit.add_argument("--source-id", default="gomes-yolo-figshare-24060960")
    gomes_audit.add_argument("--archive", type=Path, required=True)
    gomes_audit.add_argument("--classes", type=Path, required=True)
    gomes_audit.add_argument("--output", type=Path, default=DEFAULT_INTAKE)

    gomes_family = commands.add_parser(
        "real-gomes-family-audit", help="Audit overlap across Gomes public releases"
    )
    gomes_family.add_argument("--semantic-archive", type=Path, required=True)
    gomes_family.add_argument("--yolo-root", type=Path, required=True)
    gomes_family.add_argument("--classes", type=Path, required=True)
    gomes_family.add_argument("--output", type=Path, required=True)

    download = commands.add_parser(
        "real-source-download", help="Download a pinned approved public artifact"
    )
    download.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    download.add_argument("--source-id", required=True)
    download.add_argument("--output", type=Path, default=DEFAULT_DOWNLOADS)

    audit = commands.add_parser(
        "real-dataset-audit", help="Audit licensed real-domain sources and splits"
    )
    _add_dataset_inputs(audit)

    materialize = commands.add_parser(
        "real-dataset-materialize", help="Create an immutable real-domain dataset"
    )
    _add_dataset_inputs(materialize)
    materialize.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    materialize.add_argument("--dataset-id", required=True)
    materialize.add_argument("--dataset-version", default="real-domain-v1")

    inspect = commands.add_parser(
        "real-dataset-inspect", help="Inspect a materialized real-domain identity"
    )
    inspect.add_argument("--input", type=Path, default=DEFAULT_DATASET)

    view = commands.add_parser(
        "real-training-view-materialize",
        help="Create the deterministic real/synthetic YOLO11n v3 view",
    )
    view.add_argument("--real-dataset", type=Path, default=DEFAULT_DATASET)
    view.add_argument(
        "--synthetic-view", type=Path, default=Path("data/research/visual_yolo_v2")
    )
    view.add_argument("--output", type=Path, default=DEFAULT_V3_VIEW)
    view.add_argument("--seed", type=int, default=7)

    regression = commands.add_parser(
        "real-synthetic-regression", help="Evaluate v3 on synthetic regression data"
    )
    regression.add_argument("--model", type=Path, required=True)
    regression.add_argument("--dataset", type=Path, default=DEFAULT_V3_VIEW)
    regression.add_argument("--real-validation", type=Path, required=True)
    regression.add_argument("--output", type=Path, required=True)
    regression.add_argument("--device", default="mps")
    regression.add_argument("--imgsz", type=int, default=640)

    candidate = commands.add_parser(
        "real-validation-gate", help="Freeze the v3 real and regression gates"
    )
    candidate.add_argument("--real-validation", type=Path, required=True)
    candidate.add_argument("--synthetic-regression", type=Path, required=True)
    candidate.add_argument("--v2-synthetic-baseline", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)

    blind = commands.add_parser(
        "real-blind-materialize", help="Unlock the real blind view after package gate"
    )
    blind.add_argument("--real-dataset", type=Path, default=DEFAULT_DATASET)
    blind.add_argument("--package", type=Path, required=True)
    blind.add_argument("--output", type=Path, required=True)


def handle_real_domain_command(args):
    handlers = {
        "real-source-audit": lambda: audit_real_sources(
            args.registry, args.output, args.content_audits
        ),
        "real-source-intake": lambda: intake_real_source(
            args.registry, args.source_id, args.output, sample_limit=args.sample_limit
        ),
        "real-source-export-audit": lambda: audit_source_export(
            args.registry, args.source_id, args.archive, args.output
        ),
        "real-source-content-audit": lambda: audit_source_content(
            args.registry, args.source_id, args.archive, args.output
        ),
        "real-gomes-yolo-audit": lambda: audit_gomes_yolo_component(
            args.registry, args.source_id, args.archive, args.classes, args.output
        ),
        "real-gomes-family-audit": lambda: audit_gomes_source_family(
            args.semantic_archive, args.yolo_root, args.classes, args.output
        ),
        "real-source-download": lambda: download_approved_source(
            args.registry, args.source_id, args.output
        ),
        "real-dataset-audit": lambda: _audit_dataset(args),
        "real-dataset-materialize": lambda: materialize_real_dataset(
            args.registry,
            args.manifest,
            args.source_root,
            args.output,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
        ),
        "real-dataset-inspect": lambda: inspect_real_dataset(args.input),
        "real-training-view-materialize": lambda: materialize_real_training_view(
            args.real_dataset, args.synthetic_view, args.output, seed=args.seed
        ),
        "real-synthetic-regression": lambda: evaluate_synthetic_regression(
            args.model,
            args.dataset,
            args.real_validation,
            args.output,
            device=args.device,
            imgsz=args.imgsz,
        ),
        "real-validation-gate": lambda: validate_v3_candidate(
            args.real_validation,
            args.synthetic_regression,
            args.v2_synthetic_baseline,
            args.output,
        ),
        "real-blind-materialize": lambda: materialize_real_blind_view(
            args.real_dataset, args.package, args.output
        ),
    }
    handler = handlers.get(args.command)
    return (False, None) if handler is None else (True, handler())


def _add_dataset_inputs(parser):
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)


def _audit_dataset(args):
    result = audit_real_dataset(args.registry, args.manifest, args.source_root)
    result.pop("rows", None)
    return result
