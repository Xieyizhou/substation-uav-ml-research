"""Inspect and validate frozen visual research identities and templates."""

from __future__ import annotations

import argparse
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
        else:
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"Visual command failed: {error}")
        return 1
