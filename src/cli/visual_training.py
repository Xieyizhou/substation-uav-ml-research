"""CLI boundary for visual training, export, evaluation, and packaging."""

from __future__ import annotations

from pathlib import Path

from src.ml.visual_heldout_view import materialize_heldout_view
from src.ml.visual_onnx_gate import validate_onnx_equivalence
from src.ml.visual_training_view import materialize_training_view
from src.ml.visual_static_replay import (
    materialize_static_replay,
    run_static_replay,
)
from src.ml.visual_yolo_evaluation import evaluate_yolo
from src.ml.visual_yolo_package import export_yolo_package, validate_yolo_package
from src.ml.visual_yolo_training import train_yolo


DEFAULT_COLLECTION = Path("data/research/visual_collection_v1")
DEFAULT_TRAINING_VIEW = Path("data/research/visual_yolo_v1")
DEFAULT_CONFIG = Path("config/perception/visual_yolo11n_baseline.json")
DEFAULT_RUN = Path("models/equipment/visual-yolo11n-baseline-v1")


def add_training_parsers(commands):
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


def handle_training_command(args):
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
    if args.command == "static-replay-materialize":
        return materialize_static_replay(
            args.package, args.dataset, args.benchmark, args.output
        )
    if args.command == "static-replay-run":
        return run_static_replay(args.input, args.package, args.dataset)
    return None
