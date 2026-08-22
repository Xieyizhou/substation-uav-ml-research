"""CLI boundary for temporal, real-image, and latency diagnostics."""

from pathlib import Path

from src.vision.evaluation.onnx_latency_probe import probe_onnx_latency
from src.vision.evaluation.reality_stress import (
    evaluate_reality_stress,
    materialize_reality_stress,
)
from src.vision.evaluation.temporal_observations import (
    evaluate_temporal_jsonl,
    evaluate_temporal_replay,
)


def add_diagnostic_parsers(commands):
    temporal = commands.add_parser(
        "temporal-observations",
        help="Apply frozen temporal confirmation to ordered prediction JSONL",
    )
    temporal.add_argument("--input", type=Path, required=True)
    temporal.add_argument("--output", type=Path, required=True)
    temporal.add_argument("--threshold", type=float, default=0.05)

    replay = commands.add_parser(
        "temporal-replay-evaluate",
        help="Evaluate saved static replay predictions on the full timeline",
    )
    replay.add_argument("--predictions", type=Path, required=True)
    replay.add_argument("--membership", type=Path, required=True)
    replay.add_argument("--dataset", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--threshold", type=float, default=0.37)
    replay.add_argument("--source-rate", type=float, default=30.0)
    replay.add_argument("--imgsz", type=int, choices=[320, 640], default=320)

    materialize = commands.add_parser(
        "reality-stress-materialize",
        help="Create a provenance-bound real-image stress view",
    )
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--annotations", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser(
        "reality-stress-evaluate",
        help="Evaluate a frozen model on a real-image stress view",
    )
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--imgsz", type=int, choices=[320, 640], default=640)
    evaluate.add_argument("--threshold", type=float, default=0.37)

    latency = commands.add_parser(
        "onnx-latency-probe",
        help="Measure first-use and sustained ONNX inference latency",
    )
    latency.add_argument("--model", type=Path, required=True)
    latency.add_argument("--image", type=Path, required=True)
    latency.add_argument("--output", type=Path, required=True)
    latency.add_argument("--device", default="cpu")
    latency.add_argument("--imgsz", type=int, choices=[320, 416, 640], default=416)
    latency.add_argument("--warm-iterations", type=int, default=200)


def handle_diagnostic_command(args):
    if args.command == "temporal-observations":
        return True, evaluate_temporal_jsonl(
            args.input, args.output, threshold=args.threshold
        )
    if args.command == "temporal-replay-evaluate":
        return True, evaluate_temporal_replay(
            args.predictions,
            args.membership,
            args.dataset,
            args.output,
            threshold=args.threshold,
            source_rate_hz=args.source_rate,
            input_size=args.imgsz,
        )
    if args.command == "reality-stress-materialize":
        return True, materialize_reality_stress(
            args.source, args.annotations, args.output
        )
    if args.command == "reality-stress-evaluate":
        return True, evaluate_reality_stress(
            args.model,
            args.dataset,
            args.output,
            device=args.device,
            imgsz=args.imgsz,
            threshold=args.threshold,
        )
    if args.command == "onnx-latency-probe":
        return True, probe_onnx_latency(
            args.model,
            args.image,
            args.output,
            device=args.device,
            imgsz=args.imgsz,
            warm_iterations=args.warm_iterations,
        )
    return False, None
