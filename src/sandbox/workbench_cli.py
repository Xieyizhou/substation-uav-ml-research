"""CLI boundary for visual model workbench datasets and runs."""

from __future__ import annotations

from pathlib import Path


def register_workbench_commands(commands):
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


def _mapping(values):
    result = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator:
            raise ValueError("class map must use SOURCE_ID=CANONICAL_NAME")
        result[int(source)] = target
    return result


def _overrides(args):
    names = ("epochs", "patience", "imgsz", "batch", "workers", "device")
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def handle_workbench_command(args, config):
    if not str(args.command).startswith("workbench-"):
        return None
    if config.profile != "development":
        raise ValueError("model workbench is available only in development profile")
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
    raise ValueError("unsupported workbench command")
