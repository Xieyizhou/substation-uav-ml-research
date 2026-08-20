"""Idempotent study scheduling and local artifact ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.domain_randomization import (
    load_ranges,
    sample_manifest,
)
from src.maps.map_catalog import map_by_id, spawn_pose_text
from src.study.matrix import tier_matrix
from src.study.capability_scenario import materialize_reachable_scenario
from src.study.dynamic_replanning import materialize_dynamic_scenario


ROOT = Path(__file__).resolve().parents[2]
RANDOMIZATION_CONFIG = ROOT / "config/perception/domain_randomization.json"


def result_root(results_dir, study_id, tier):
    return Path(results_dir) / study_id / tier / "results"


def schedule_tier(registry, study_id, tier):
    study = registry.get_study(study_id)
    candidate = registry.get_model(study["candidate_model"])
    matrix = tier_matrix(tier, include_champion=bool(study.get("champion_model")))
    return registry.ensure_runs(
        study_id,
        tier,
        matrix,
        config_hash=object_sha256({"tier": tier, "matrix": matrix}),
        model_hash=candidate["onnx_hash"],
    )


def _flight_arguments(
    condition, model_path, scenario_manifest, oracle_planner_config=None,
    dynamic_scenario=None, max_speed_m_s=None,
):
    common = [
        "--scenario-manifest",
        str(scenario_manifest),
        "--sensor-startup-timeout", "20", "--sensor-stale-after", "2.0",
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
        arguments = [
            "--enable-perception",
            "--perception-source",
            "gazebo_lidar_2d",
            "--risk-model",
            "geometric",
            *common,
        ]
        if dynamic_scenario is not None:
            arguments.extend([
                "--dynamic-replan-scenario", str(dynamic_scenario),
                "--return-home", "--replan-risk-level", "warning",
                "--max-replans", "1", "--detection-fov", "360",
                "--detection-range", "4", "--warning-distance", "2.5",
            ])
        if max_speed_m_s is not None:
            arguments.extend([
                "--max-speed", str(float(max_speed_m_s)),
                "--return-speed-scale", "1.0",
            ])
        return arguments
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


def write_run_queue(registry, study_id, tier, output_dir):
    """Write an auditable worker queue without embedding machine-specific state in Git."""
    study = registry.get_study(study_id)
    model = registry.get_model(study["candidate_model"])
    model_path = Path(model["path"]) / "model.onnx"
    output_dir = Path(output_dir) / study_id / tier
    scenarios_dir = output_dir / "scenarios"
    worlds_dir = output_dir / "worlds"
    randomization = load_ranges(RANDOMIZATION_CONFIG)
    definitions = {
        (row["scenario_id"], row["condition"]): row
        for row in tier_matrix(tier, include_champion=bool(study.get("champion_model")))
    }
    dynamic_tiers = {"dynamic-replanning", "speed-envelope"}
    rows = []
    for run in schedule_tier(registry, study_id, tier):
        definition = definitions[(run["scenario_id"], run["condition"])]
        scenario_profile = definition.get("scenario_profile")
        scenario_manifest = scenarios_dir / f"{run['scenario_id']}.json"
        world_path = worlds_dir / f"{run['scenario_id']}.sdf"
        oracle_planner = worlds_dir / f"{run['scenario_id']}.planner.json"
        dynamic_scenario = scenarios_dir / f"{run['scenario_id']}.dynamic.json"
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
                materialize_reachable_scenario(
                    manifest,
                    entry,
                    run["target_id"],
                    world_path,
                    scenario_manifest,
                    oracle_planner,
                    scenario_profile,
                )
        if tier in dynamic_tiers and not dynamic_scenario.is_file():
            materialize_dynamic_scenario(
                run["map_id"],
                definition["injection_phase"],
                dynamic_scenario,
                planner_path=oracle_planner,
            )
        rows.append(
            {
                "run_id": run["run_id"],
                "scenario_id": run["scenario_id"],
                "condition": run["condition"],
                "map_id": run["map_id"],
                "target_id": run["target_id"],
                "seed": run["seed"],
                "scenario_profile": scenario_profile,
                "required_capabilities": list(
                    definition.get("required_capabilities", ())
                ),
                "injection_phase": definition.get("injection_phase"),
                "representative_scenario": definition.get(
                    "representative_scenario"
                ),
                "speed_m_s": definition.get("speed_m_s"),
                "repeat": definition.get("repeat"),
                "dynamic_replan_scenario": (
                    str(dynamic_scenario)
                    if tier in dynamic_tiers
                    else None
                ),
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
                        dynamic_scenario if tier in dynamic_tiers else None,
                        definition.get("speed_m_s"),
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
