"""Train, package, predict, evaluate, and benchmark LiDAR models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml.dataset import load_dataset
from src.ml.model_package import create_model_package, validate_model_package
from src.ml.predictions import evaluate_predictions, predict_dataset, read_predictions
from src.ml.protocol import experiment_matrix, load_protocol
from src.ml.train_lidar import train as train_lidar


def build_parser():
    parser = argparse.ArgumentParser(description="ML research model tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train and export a LiDAR model")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--model-id")
    train.add_argument("--parent-model")
    package = commands.add_parser("package", help="Create a versioned model package")
    package.add_argument("--model", type=Path, required=True)
    package.add_argument("--dataset-manifest", type=Path, required=True)
    package.add_argument("--history", type=Path, required=True)
    package.add_argument("--metrics", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--model-id")
    package.add_argument("--parent-model")
    predict = commands.add_parser("predict", help="Run ONNX predictions on a split")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--dataset", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    evaluate = commands.add_parser("evaluate", help="Evaluate saved predictions")
    evaluate.add_argument("--predictions", type=Path)
    evaluate.add_argument("--model", type=Path)
    evaluate.add_argument("--dataset", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    benchmark = commands.add_parser("benchmark", help="Summarize inference latency")
    benchmark.add_argument("--predictions", type=Path)
    benchmark.add_argument("--model", type=Path)
    benchmark.add_argument("--dataset", type=Path)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    dataset = commands.add_parser("dataset", help="Validate legacy dataset JSONL")
    dataset.add_argument("--input", type=Path, required=True)
    protocol = commands.add_parser("protocol", help="Validate a formal experiment matrix")
    protocol.add_argument("--config", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Validate a model package")
    inspect.add_argument("--package", type=Path, required=True)
    return parser


def _train(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch size, and patience must be positive")
    if args.output.suffix:
        raise ValueError("--output must be a model package directory, not a file")
    resolved_model_id = args.model_id or args.output.name
    result = train_lidar(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        patience=args.patience,
        model_id=resolved_model_id,
    )
    manifest_path = args.dataset.parent / "dataset_manifest.json"
    manifest = create_model_package(
        args.output,
        onnx_path=result["model"],
        dataset_manifest_path=manifest_path,
        training_history_path=result["history"],
        offline_metrics_path=result["metrics"],
        model_id=resolved_model_id,
        parent_model=args.parent_model,
        training_parameters={
            "epochs_requested": args.epochs,
            "epochs_completed": result["epochs_completed"],
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "seed": args.seed,
        },
    )
    return {"package": str(args.output), "model_id": manifest["model_id"]}


def _prediction_input(args):
    if args.predictions:
        return args.predictions
    if not args.model or not args.dataset:
        raise ValueError("provide --predictions or both --model and --dataset")
    if not args.output:
        raise ValueError("--output is required when generating predictions")
    model = args.model / "model.onnx" if args.model.is_dir() else args.model
    return predict_dataset(model, args.dataset, args.output, split=args.split)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = _train(args)
        elif args.command == "package":
            result = create_model_package(
                args.output,
                onnx_path=args.model,
                dataset_manifest_path=args.dataset_manifest,
                training_history_path=args.history,
                offline_metrics_path=args.metrics,
                model_id=args.model_id,
                parent_model=args.parent_model,
            )
        elif args.command == "predict":
            model = args.model / "model.onnx" if args.model.is_dir() else args.model
            result = {
                "predictions": str(
                    predict_dataset(
                        model, args.dataset, args.output, split=args.split
                    )
                )
            }
        elif args.command == "evaluate":
            result = evaluate_predictions(_prediction_input(args))
        elif args.command == "benchmark":
            result = evaluate_predictions(_prediction_input(args))["latency"]
        elif args.command == "dataset":
            samples = load_dataset(args.input)
            result = {
                "samples": len(samples),
                "splits": {
                    split: sum(sample.split == split for sample in samples)
                    for split in ("train", "validation", "test")
                },
            }
        elif args.command == "protocol":
            protocol = load_protocol(args.config)
            result = {"runs": len(experiment_matrix(protocol))}
        elif args.command == "inspect":
            result = validate_model_package(args.package)
        else:
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Model command failed: {error}")
        return 1
