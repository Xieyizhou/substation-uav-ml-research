"""Allow-listed visual, LiDAR, and acceptance workflow commands."""

from __future__ import annotations

from datetime import datetime, timezone
import sys

from src.sandbox.command_models import SandboxCommand
from src.sandbox.experiment_recipe import materialize_recipe
from src.sandbox.lidar_jobs import build_lidar_command
from src.sandbox.workflow import artifact_reference


def build_visual_command(config, parameters):
    if not isinstance(parameters, dict):
        raise ValueError("experiment parameters must be an object")
    allowed = {"name", "partition", "input_size", "frame_skip_interval", "frame_limit"}
    if set(parameters) - allowed:
        raise ValueError("unsupported experiment parameter")
    recipe, output = materialize_recipe(
        config.project_root,
        parameters.get("name"),
        partition=parameters.get("partition", "validation"),
        input_size=parameters.get("input_size", 416),
        frame_skip_interval=parameters.get("frame_skip_interval", 1),
        frame_limit=parameters.get("frame_limit"),
    )
    timeout = 900.0 if recipe.source_frame_count <= 256 else (
        1_800.0 if recipe.source_frame_count <= 1_024 else 7_200.0
    )
    return SandboxCommand(
        "experiment-run",
        (
            sys.executable, "main.py", "sandbox", "--project-root",
            str(config.project_root), "experiment-run", "--recipe",
            str(output / "recipe.json"),
        ),
        timeout,
        workflow="visual_replay",
        artifacts=(
            artifact_reference(
                config.project_root, "visual_recipe",
                recipe.recipe_identity_sha256, output / "recipe.json",
            ),
            artifact_reference(
                config.project_root, "visual_model_package",
                recipe.package_identity_sha256,
                config.project_root / recipe.package_root,
            ),
            artifact_reference(
                config.project_root, "visual_membership",
                recipe.membership_sha256,
                config.project_root / recipe.dataset_root,
            ),
        ),
        expected_outputs=(
            (output / "result.json").relative_to(config.project_root).as_posix(),
        ),
        requires_runtime_idle=False,
    )


def build_lidar_sandbox_command(config, action):
    name, argv, timeout, artifacts, outputs, needs_idle = build_lidar_command(
        config, action
    )
    workflow = "lidar_replay" if action == "lidar-replay-gate" else "lidar_closed_loop"
    return SandboxCommand(
        name, argv, timeout, workflow=workflow, artifacts=artifacts,
        expected_outputs=outputs, requires_runtime_idle=needs_idle,
    )


def _acceptance_command(config):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = config.project_root / "outputs/sandbox/acceptance" / stamp
    return SandboxCommand(
        "sandbox-acceptance",
        (
            sys.executable, "main.py", "sandbox", "--project-root",
            str(config.project_root), "acceptance-run", "--output", str(output),
        ),
        120.0,
        workflow="multimodal_acceptance",
        expected_outputs=(
            (output / "acceptance.json").relative_to(config.project_root).as_posix(),
        ),
        requires_runtime_idle=False,
    )


def build_workflow_command(config, parameters):
    if not isinstance(parameters, dict):
        raise ValueError("workflow parameters must be an object")
    workflow = parameters.get("workflow")
    if workflow == "visual_replay":
        return build_visual_command(
            config, {key: value for key, value in parameters.items() if key != "workflow"}
        )
    if set(parameters) != {"workflow"}:
        raise ValueError("unsupported workflow parameter")
    if workflow == "lidar_replay":
        return build_lidar_sandbox_command(config, "lidar-replay-gate")
    if workflow == "lidar_closed_loop":
        return build_lidar_sandbox_command(config, "lidar-closed-loop-next")
    if workflow == "multimodal_acceptance":
        return _acceptance_command(config)
    raise ValueError("unsupported sandbox workflow")
