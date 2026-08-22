"""CLI boundary for visual training, export, evaluation, and packaging."""

from __future__ import annotations

from pathlib import Path

from src.cli.visual_real_domain import (
    add_real_domain_parsers,
    handle_real_domain_command,
)
from src.cli.visual_diagnostics import (
    add_diagnostic_parsers,
    handle_diagnostic_command,
)

from src.vision.evaluation.heldout_view import materialize_heldout_view
from src.vision.evaluation.paired_heldout import (
    evaluate_paired_heldout,
    materialize_paired_heldout_view,
)
from src.vision.evaluation.onnx_gate import validate_onnx_equivalence
from src.vision.training.view import materialize_training_view
from src.vision.replay.static_replay import (
    materialize_static_replay,
    run_static_replay,
)
from src.vision.evaluation.yolo_evaluation import evaluate_yolo
from src.vision.evaluation.yolo_package import export_yolo_package, validate_yolo_package
from src.vision.training.yolo_training import train_yolo


DEFAULT_COLLECTION = Path("data/research/visual_collection_v1")
DEFAULT_TRAINING_VIEW = Path("data/research/visual_yolo_v1")
DEFAULT_CONFIG = Path("config/perception/visual_yolo11n_baseline.json")
DEFAULT_RUN = Path("models/equipment/visual-yolo11n-baseline-v1")


def add_training_parsers(commands):
    add_real_domain_parsers(commands)
    add_diagnostic_parsers(commands)

    view = commands.add_parser(
        "training-view-materialize",
        help="Create a deterministic split-safe YOLO training view",
    )
    view.add_argument("--collection-root", type=Path, default=DEFAULT_COLLECTION)
    view.add_argument("--output", type=Path, default=DEFAULT_TRAINING_VIEW)
    view.add_argument("--seed", type=int, default=7)

    train = commands.add_parser(
        "train-yolo", help="Train the frozen YOLO11n visual baseline"
    )
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--dataset", type=Path, default=DEFAULT_TRAINING_VIEW)
    train.add_argument("--output", type=Path, default=DEFAULT_RUN)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--smoke", action="store_true")

    evaluate = commands.add_parser(
        "evaluate-yolo", help="Evaluate a visual YOLO model on an allowed view"
    )
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--partition",
        choices=["validation", "full_validation", "heldout_test"],
        default="full_validation",
    )
    evaluate.add_argument("--device", default="mps")
    evaluate.add_argument("--imgsz", type=int, default=640)

    export = commands.add_parser(
        "export-yolo", help="Export 320/416/640 ONNX model package variants"
    )
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--training-view-identity", type=Path, required=True)
    export.add_argument("--training-provenance", type=Path, required=True)
    export.add_argument("--validation-results", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--equivalence-dataset", type=Path, default=DEFAULT_TRAINING_VIEW
    )
    export.add_argument("--equivalence-device", default="cpu")
    export.add_argument(
        "--v3-gate", type=Path,
        help="Passing real-domain v3 gate to bind into the package",
    )

    inspect = commands.add_parser(
        "model-package-inspect", help="Validate a visual YOLO model package"
    )
    inspect.add_argument("--input", type=Path, required=True)

    gate = commands.add_parser(
        "onnx-equivalence",
        help="Run the fixed 200-frame PT/ONNX equivalence gate",
    )
    gate.add_argument("--pt-model", type=Path, required=True)
    gate.add_argument("--onnx-model", type=Path, required=True)
    gate.add_argument("--dataset", type=Path, default=DEFAULT_TRAINING_VIEW)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--imgsz", type=int, default=640)
    gate.add_argument("--device", default="cpu")

    heldout = commands.add_parser(
        "heldout-view-materialize",
        help="Unlock held-out labels for one already-frozen model package",
    )
    heldout.add_argument("--collection-root", type=Path, default=DEFAULT_COLLECTION)
    heldout.add_argument("--package", type=Path, required=True)
    heldout.add_argument("--output", type=Path, required=True)

    paired = commands.add_parser(
        "paired-heldout-materialize",
        help="Unlock one v1/v2 comparison on the frozen v2 blind dataset",
    )
    paired.add_argument("--collection-root", type=Path, required=True)
    paired.add_argument("--v1-package", type=Path, required=True)
    paired.add_argument("--v2-package", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)

    paired_evaluate = commands.add_parser(
        "paired-heldout-evaluate",
        help="Run the identity-bound v1/v2 blind comparison exactly once",
    )
    paired_evaluate.add_argument("--dataset", type=Path, required=True)
    paired_evaluate.add_argument("--output", type=Path, required=True)
    paired_evaluate.add_argument("--device", default="cpu")

    static_materialize = commands.add_parser(
        "static-replay-materialize",
        help="Bind the nine static replay templates to frozen identities",
    )
    static_materialize.add_argument("--package", type=Path, required=True)
    static_materialize.add_argument("--dataset", type=Path, required=True)
    static_materialize.add_argument(
        "--benchmark", type=Path, default=Path("benchmarks/visual_static_v1")
    )
    static_materialize.add_argument("--output", type=Path, required=True)

    static_run = commands.add_parser(
        "static-replay-run", help="Execute all incomplete static replay conditions"
    )
    static_run.add_argument("--input", type=Path, required=True)
    static_run.add_argument("--package", type=Path, required=True)
    static_run.add_argument("--dataset", type=Path, required=True)
    static_run.add_argument(
        "--condition",
        dest="condition_ids",
        action="append",
        default=[],
        help="Run only this materialized condition; repeat to select more",
    )

def handle_training_command(args):
    handled, result = handle_real_domain_command(args)
    if handled:
        return result
    handled, result = handle_diagnostic_command(args)
    if handled:
        return result
    if args.command == "training-view-materialize":
        return materialize_training_view(
            args.collection_root, args.output, seed=args.seed
        )
    if args.command == "train-yolo":
        return train_yolo(
            args.config,
            args.dataset,
            args.output,
            resume=args.resume,
            smoke=args.smoke,
        )
    if args.command == "evaluate-yolo":
        return evaluate_yolo(
            args.model,
            args.dataset,
            args.output,
            partition=args.partition,
            device=args.device,
            imgsz=args.imgsz,
        )
    if args.command == "export-yolo":
        return export_yolo_package(
            args.weights,
            args.training_view_identity,
            args.training_provenance,
            args.validation_results,
            args.output,
            equivalence_dataset=args.equivalence_dataset,
            equivalence_device=args.equivalence_device,
            v3_gate_path=args.v3_gate,
        )
    if args.command == "model-package-inspect":
        return validate_yolo_package(args.input)
    if args.command == "onnx-equivalence":
        return validate_onnx_equivalence(
            args.pt_model,
            args.onnx_model,
            args.dataset,
            args.output,
            imgsz=args.imgsz,
            device=args.device,
        )
    if args.command == "heldout-view-materialize":
        return materialize_heldout_view(
            args.collection_root, args.package, args.output
        )
    if args.command == "paired-heldout-materialize":
        return materialize_paired_heldout_view(
            args.collection_root, args.v1_package, args.v2_package, args.output
        )
    if args.command == "paired-heldout-evaluate":
        return evaluate_paired_heldout(args.dataset, args.output, device=args.device)
    if args.command == "static-replay-materialize":
        return materialize_static_replay(
            args.package, args.dataset, args.benchmark, args.output
        )
    if args.command == "static-replay-run":
        return run_static_replay(
            args.input,
            args.package,
            args.dataset,
            condition_ids=args.condition_ids,
        )
    return None
