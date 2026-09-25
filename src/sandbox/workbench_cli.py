"""CLI boundary for visual model workbench datasets and runs."""

from __future__ import annotations

from pathlib import Path


def register_workbench_commands(commands):
    feedback = commands.add_parser("workbench-feedback-register", help="Register explicitly reviewed feedback with fixed base validation")
    feedback.add_argument("--dataset-id", required=True)
    feedback.add_argument("--base-dataset-id", required=True)
    feedback.add_argument("--collection-id", action="append", required=True)
    feedback.add_argument("--base-train-limit", type=int, choices=(128, 256, 512), default=256)
    feedback.add_argument("--validation-limit", type=int, choices=(64, 128, 256), default=64)
    imported = commands.add_parser(
        "workbench-dataset-import", help="Audit and import a YOLO Detect dataset"
    )
    imported.add_argument("--source", type=Path, required=True)
    imported.add_argument("--dataset-id", required=True)
    imported.add_argument("--class-map", action="append", default=[])
    recipe = commands.add_parser(
        "workbench-recipe-create", help="Create a workbench training recipe"
    )
    recipe.add_argument("--experiment-id", required=True)
    recipe.add_argument("--dataset-id", required=True)
    recipe.add_argument("--preset", choices=["smoke", "quick", "full"], required=True)
    for name in ("epochs", "patience", "imgsz", "batch", "workers"):
        recipe.add_argument(f"--{name}", type=int)
    recipe.add_argument("--device", choices=["auto", "mps", "cpu"])
    recipe.add_argument("--freeze", type=int, choices=(0, 5, 10),
                        help="Freeze the first N model layers during a new fine-tuning run")
    source = recipe.add_mutually_exclusive_group()
    source.add_argument("--initial-weights", type=Path,
                        help="Pin a local historical .pt checkpoint for a new training run")
    source.add_argument("--parent-experiment-id",
                        help="Fine-tune a verified workbench model")
    for command, help_text in (
        ("workbench-run", "Run a workbench recipe"),
        ("workbench-resume", "Resume a workbench recipe"),
    ):
        parser = commands.add_parser(command, help=help_text)
        parser.add_argument("--recipe", type=Path, required=True)
    for command, help_text in (
        ("workbench-validate", "Re-run validation for an existing workbench run"),
        ("workbench-replay", "Re-run fixed validation replay for an existing run"),
    ):
        parser = commands.add_parser(command, help=help_text)
        parser.add_argument("--input", type=Path, required=True)
    inspect = commands.add_parser("workbench-inspect", help="Inspect a workbench run")
    inspect.add_argument("--input", type=Path, required=True)
    inference = commands.add_parser(
        "workbench-image-infer", help="Run a verified ONNX model on one managed image"
    )
    inference.add_argument("--runs-root", type=Path, required=True)
    inference.add_argument("--experiment-id", required=True)
    inference.add_argument("--comparison-experiment-id")
    inference.add_argument("--source", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)


def _mapping(values):
    result = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator:
            raise ValueError("class map must use SOURCE_ID=CANONICAL_NAME")
        result[int(source)] = target
    return result


def _overrides(args):
    names = ("epochs", "patience", "imgsz", "batch", "workers", "device", "freeze")
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def handle_workbench_command(args, config):
    if not str(args.command).startswith("workbench-"):
        return None
    if config.profile != "development":
        raise ValueError("model workbench is available only in development profile")
    if args.command == "workbench-feedback-register":
        from src.sandbox.feedback_dataset import register_feedback_dataset
        return register_feedback_dataset(config, args.dataset_id, args.base_dataset_id, args.collection_id,
                                         base_train_limit=args.base_train_limit, validation_limit=args.validation_limit)
    if args.command == "workbench-dataset-import":
        from src.sandbox.workbench_datasets import import_yolo_dataset
        return import_yolo_dataset(
            args.source, config.workbench_datasets_root, args.dataset_id,
            _mapping(args.class_map) or None,
        )
    if args.command == "workbench-recipe-create":
        from src.sandbox.workbench_recipe import materialize_workbench_recipe
        recipe, root = materialize_workbench_recipe(
            config.project_root, config.workbench_datasets_root,
            config.workbench_runs_root, args.experiment_id, args.dataset_id,
            args.preset, _overrides(args),
            initial_weights=args.initial_weights,
            parent_experiment_id=args.parent_experiment_id,
        )
        return {"path": str(root / "recipe.json"), "recipe": recipe.to_record()}
    if args.command in ("workbench-run", "workbench-resume"):
        from src.sandbox.workbench_runner import run_workbench_experiment
        return run_workbench_experiment(
            config.project_root, config.workbench_datasets_root,
            config.workbench_runs_root, args.recipe,
            resume=args.command == "workbench-resume",
        )
    if args.command == "workbench-validate":
        from src.sandbox.workbench_lifecycle import validate_workbench_run
        return validate_workbench_run(args.input)
    if args.command == "workbench-replay":
        from src.sandbox.workbench_lifecycle import replay_workbench_run
        return replay_workbench_run(config.project_root, args.input)
    if args.command == "workbench-inspect":
        from src.sandbox.workbench_lifecycle import inspect_workbench_run
        return inspect_workbench_run(args.input)
    if args.command == "workbench-image-infer":
        from src.sandbox.workbench_inference import run_image_inference
        return run_image_inference(
            args.runs_root, args.experiment_id, args.source, args.output,
            args.comparison_experiment_id,
        )
    raise ValueError("unsupported workbench command")
