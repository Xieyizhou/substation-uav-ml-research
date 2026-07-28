"""Train, validate, and benchmark optional research models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml import RISK_LABELS
from src.ml.dataset import load_dataset
from src.ml.metrics import (
    binary_iou,
    classification_report,
    expected_calibration_error,
    latency_summary,
)
from src.ml.domain_randomization import load_ranges, sample_manifest
from src.ml.protocol import experiment_matrix, load_protocol
from src.ml.train_lidar import train as train_lidar


def build_parser():
    parser = argparse.ArgumentParser(description="ML research model tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a LiDAR or equipment model")
    train.add_argument("--kind", choices=["lidar", "equipment"], required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--base-weights", default="yolo11n.pt")
    evaluate = commands.add_parser("evaluate", help="Evaluate saved predictions")
    evaluate.add_argument("--predictions", type=Path, required=True)
    benchmark = commands.add_parser("benchmark", help="Summarize inference latency")
    benchmark.add_argument("--predictions", type=Path, required=True)
    dataset = commands.add_parser("dataset", help="Validate dataset split isolation")
    dataset.add_argument("--input", type=Path, required=True)
    protocol = commands.add_parser("protocol", help="Validate a formal experiment matrix")
    protocol.add_argument("--config", type=Path, required=True)
    randomize = commands.add_parser("randomize", help="Create a reproducible scenario manifest")
    randomize.add_argument("--config", type=Path, required=True)
    randomize.add_argument("--map", dest="map_id", required=True)
    randomize.add_argument("--seed", type=int, required=True)
    return parser


def _read_predictions(path):
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("prediction file is empty")
    return rows


def _train_equipment(args):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO training requires requirements-ml.txt") from error
    model = YOLO(args.base_weights)
    model.train(
        data=str(args.dataset),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        project=str(args.output.parent),
        name=args.output.stem,
    )
    return 0


def _evaluate(path):
    rows = _read_predictions(path)
    labels = [row["risk_label"] for row in rows]
    predictions = [row["predicted_risk"] for row in rows]
    confidences = [float(row.get("confidence", 0.0)) for row in rows]
    report = classification_report(labels, predictions, RISK_LABELS)
    report["ece"] = expected_calibration_error(labels, confidences, predictions)
    traversability_rows = [
        row
        for row in rows
        if "traversability" in row and "predicted_traversability" in row
    ]
    report["traversability_iou"] = (
        sum(
            binary_iou(row["traversability"], row["predicted_traversability"])
            for row in traversability_rows
        )
        / len(traversability_rows)
        if traversability_rows
        else None
    )
    report["latency"] = latency_summary(
        [float(row["latency_ms"]) for row in rows if "latency_ms" in row]
    )
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dataset":
            samples = load_dataset(args.input)
            counts = {
                split: sum(sample.split == split for sample in samples)
                for split in ("train", "validation", "test")
            }
            print(json.dumps({"samples": len(samples), "splits": counts}, indent=2))
            return 0
        if args.command == "protocol":
            protocol = load_protocol(args.config)
            matrix = experiment_matrix(protocol)
            print(
                json.dumps(
                    {
                        "runs": len(matrix),
                        "maps": len(protocol["maps"]),
                        "targets": len(protocol["targets"]),
                        "conditions": len(protocol["conditions"]),
                        "seeds_per_condition": len(protocol["seeds"]),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "randomize":
            config = load_ranges(args.config)
            print(
                json.dumps(
                    sample_manifest(config, map_id=args.map_id, seed=args.seed),
                    indent=2,
                )
            )
            return 0
        if args.command == "train":
            if args.epochs <= 0 or args.batch_size <= 0:
                raise ValueError("epochs and batch size must be positive")
            if args.kind == "equipment":
                return _train_equipment(args)
            result = train_lidar(
                args.dataset,
                args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
            return 0
        if args.command == "evaluate":
            return _evaluate(args.predictions)
        if args.command == "benchmark":
            rows = _read_predictions(args.predictions)
            print(
                json.dumps(
                    latency_summary(
                        [
                            float(row["latency_ms"])
                            for row in rows
                            if "latency_ms" in row
                        ]
                    ),
                    indent=2,
                )
            )
            return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Model command failed: {error}")
        return 1
    return 2
