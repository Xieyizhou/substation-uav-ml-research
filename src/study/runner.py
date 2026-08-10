"""Idempotent study scheduling and local artifact ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.domain_randomization import (
    load_ranges,
    materialize_planner_config,
    materialize_world,
    sample_manifest,
)
from src.maps.map_catalog import map_by_id, project_path, spawn_pose_text
from src.planner.astar_grid import astar
from src.planner.obstacle_config import build_obstacle_map
from src.study.matrix import tier_matrix


ROOT = Path(__file__).resolve().parents[2]
RANDOMIZATION_CONFIG = ROOT / "config/perception/domain_randomization.json"


def result_root(results_dir, study_id, tier):
    return Path(results_dir) / study_id / tier / "results"


def schedule_tier(registry, study_id, tier):
    study = registry.get_study(study_id)
    matrix = tier_matrix(tier, include_champion=bool(study.get("champion_model")))
    return registry.ensure_runs(
        study_id,
        tier,
        matrix,
        config_hash=object_sha256({"tier": tier, "matrix": matrix}),
        model_hash=study["candidate_model"],
    )


def _flight_arguments(
    condition, model_path, scenario_manifest, oracle_planner_config=None
):
    common = [
        "--scenario-manifest",
        str(scenario_manifest),
        "--sensor-startup-timeout", "20",
        "--enable-local-replan",
        "--replan-mode",
        "active",
    ]
    if condition == "map_oracle":
        return [
            "--enable-perception",
            "--perception-source",
            "map_baseline",
            "--obstacle-config",
            str(oracle_planner_config),
            *common,
        ]
    if condition == "geometric_lidar":
        return [
            "--enable-perception",
            "--perception-source",
            "gazebo_lidar_2d",
            "--risk-model",
            "geometric",
            *common,
        ]
    fusion = "ml_only" if condition == "ml_lidar" else "safety_max"
    return [
        "--enable-perception",
        "--perception-source",
        "gazebo_lidar_2d",
        "--risk-model",
        str(model_path),
        "--risk-fusion",
        fusion,
        *common,
    ]


def _set_target_and_validate(planner_path, entry, target_id):
    planner_path = Path(planner_path)
    config = json.loads(planner_path.read_text(encoding="utf-8"))
    target = next(
        (item for item in entry["targets"] if item["id"] == target_id), None
    )
    if target is None:
        raise ValueError(f"{entry['id']} has no target {target_id!r}")
    config["goal_cell"] = list(target["cell"])
    obstacle_map = build_obstacle_map(config)
    astar(
        tuple(config["start_cell"]),
        tuple(config["goal_cell"]),
        obstacle_map["inflated_blocking_cells"],
        int(config["width"]),
        int(config["height"]),
    )
    write_json(planner_path, config)


def _rehash_manifest(manifest):
    value = {key: item for key, item in manifest.items() if key != "config_hash"}
    manifest["config_hash"] = object_sha256(value)


def _materialize_reachable_scenario(
    manifest, entry, target_id, world_path, scenario_manifest, oracle_planner
):
    adjustments = manifest.setdefault("feasibility_adjustments", [])
    relaxed_equipment = False
    while True:
        _rehash_manifest(manifest)
        materialize_world(
            project_path(entry["world_file"]),
            world_path,
            manifest,
            report_path=scenario_manifest,
        )
        report = json.loads(scenario_manifest.read_text(encoding="utf-8"))
        materialize_planner_config(
            project_path(entry["obstacle_config"]),
            oracle_planner,
            report,
        )
        try:
            _set_target_and_validate(oracle_planner, entry, target_id)
            return
        except ValueError:
            if manifest["unknown_obstacles"]:
                removed = manifest["unknown_obstacles"].pop()
                adjustments.append(f"removed_unreachable:{removed['id']}")
                continue
            if not relaxed_equipment:
                manifest["equipment_position_jitter_m"] = 0.0
                manifest["equipment_scale"] = 1.0
                adjustments.append("restored_baseline_equipment_geometry")
                relaxed_equipment = True
                continue
            raise


def write_run_queue(registry, study_id, tier, output_dir):
    """Write an auditable worker queue without embedding machine-specific state in Git."""
    study = registry.get_study(study_id)
    model = registry.get_model(study["candidate_model"])
    model_path = Path(model["path"]) / "model.onnx"
    output_dir = Path(output_dir) / study_id / tier
    scenarios_dir = output_dir / "scenarios"
    worlds_dir = output_dir / "worlds"
    randomization = load_ranges(RANDOMIZATION_CONFIG)
    rows = []
    for run in schedule_tier(registry, study_id, tier):
        scenario_manifest = scenarios_dir / f"{run['scenario_id']}.json"
        world_path = worlds_dir / f"{run['scenario_id']}.sdf"
        oracle_planner = worlds_dir / f"{run['scenario_id']}.planner.json"
        entry = map_by_id(run["map_id"])
        if not scenario_manifest.is_file() or (
            tier != "replay"
            and (not world_path.is_file() or not oracle_planner.is_file())
        ):
            manifest = sample_manifest(
                randomization, map_id=run["map_id"], seed=run["seed"]
            )
            if tier == "replay":
                write_json(scenario_manifest, manifest)
            else:
                _materialize_reachable_scenario(
                    manifest,
                    entry,
                    run["target_id"],
                    world_path,
                    scenario_manifest,
                    oracle_planner,
                )
        rows.append(
            {
                "run_id": run["run_id"],
                "scenario_id": run["scenario_id"],
                "condition": run["condition"],
                "map_id": run["map_id"],
                "target_id": run["target_id"],
                "seed": run["seed"],
                "status": run["status"],
                "scenario_manifest": str(scenario_manifest),
                "world_path": str(world_path) if tier != "replay" else None,
                "oracle_planner_config": (
                    str(oracle_planner) if tier != "replay" else None
                ),
                "launcher_environment": (
                    {
                        "MAP_ID": "custom",
                        "WORLD_NAME": entry["world_name"],
                        "WORLD_SRC": str(world_path),
                        "PX4_GZ_MODEL_POSE": spawn_pose_text(entry),
                        "SIM_MODEL": "x500_research",
                    }
                    if tier != "replay"
                    else None
                ),
                "launcher_command": (
                    ["bash", "scripts/flight/start_px4_substation.sh"]
                    if tier != "replay"
                    else None
                ),
                "setup_commands": [
                    ["python", "main.py", "map", "use", run["map_id"]],
                    [
                        "python",
                        "main.py",
                        "point",
                        "--map",
                        run["map_id"],
                        "use",
                        run["target_id"],
                    ],
                ],
                "flight_command": [
                    "python",
                    "main.py",
                    "task",
                    "run",
                    "fly_with_replan",
                    "--",
                    *_flight_arguments(
                        run["condition"],
                        model_path,
                        scenario_manifest,
                        oracle_planner,
                    ),
                ],
                "result_path": str(
                    result_root(Path(output_dir).parent.parent, study_id, tier)
                    / f"{run['scenario_id']}__{run['condition']}.json"
                ),
            }
        )
    path = output_dir / "run_queue.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "study_id": study_id,
            "tier": tier,
            "runs": rows,
        },
    )
    return path


def ingest_results(registry, study_id, tier, results_dir):
    """Import one JSON metric artifact per scenario/condition if available."""
    scheduled = schedule_tier(registry, study_id, tier)
    results_dir = Path(results_dir)
    scoped_results = result_root(results_dir, study_id, tier)
    imported = 0
    for run in scheduled:
        if run["status"] == "completed":
            continue
        path = scoped_results / f"{run['scenario_id']}__{run['condition']}.json"
        legacy_path = results_dir / path.name
        if not path.is_file() and legacy_path.is_file():
            path = legacy_path
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "run_id": run["run_id"],
                "scenario_id": run["scenario_id"],
                "condition": run["condition"],
            }
            if result.get("schema_version") != 1:
                raise ValueError("unsupported study result schema")
            mismatches = [
                key for key, value in expected.items() if result.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    "study result identity mismatch: " + ", ".join(mismatches)
                )
            registry.record_metrics(run["run_id"], result["metrics"])
            registry.record_artifact(
                run["run_id"], "result", path, file_sha256(path)
            )
            imported += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            registry.set_run_status(run["run_id"], "failed", failure_reason=str(error))
    return {
        "scheduled": len(scheduled),
        "imported": imported,
        "completed": len(
            registry.runs(study_id, tier=tier, statuses=("completed",))
        ),
        "pending": len(
            registry.runs(study_id, tier=tier, statuses=("pending", "failed", "blocked"))
        ),
        "run_queue": str(write_run_queue(registry, study_id, tier, results_dir)),
    }
