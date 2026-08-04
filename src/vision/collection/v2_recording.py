"""Scenario materialization for visual collection protocol v2."""

from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.collection.layout import build_layout_manifest, layout_obstacle_config
from src.vision.collection.route import build_visual_route
from src.vision.collection.timing import route_timing
from src.vision.collection.world import materialize_camera_model, write_layout_world


def _scenario_paths(output_root, row):
    root = Path(output_root) / "scenarios" / row["scenario_id"]
    return {
        "world": root / "world.sdf",
        "planner": root / "planner.json",
        "route": root / "route.json",
        "layout": root / "layout.json",
        "camera": root / "x500_research" / "model.sdf",
        "report": root / "scenario.json",
    }


def _write_scenario_inputs(paths, layout, route, camera_pitch_down_deg):
    write_layout_world(paths["world"], layout)
    write_json(paths["planner"], layout_obstacle_config(layout))
    write_json(paths["route"], route.to_record())
    write_json(paths["layout"], layout.to_record())
    materialize_camera_model(
        Path("simulation/models/x500_research/model.sdf"),
        paths["camera"],
        layout.camera_noise_stddev,
        pitch_down_deg=camera_pitch_down_deg,
    )


def _write_scenario_report(paths, plan, row, protocol, layout, route, timing):
    report = {
        "collection_plan_identity_sha256": plan["collection_plan_identity_sha256"],
        "collection_protocol_id": protocol.protocol_id,
        "protocol_identity_sha256": protocol.identity_sha256,
        "scenario_id": row["scenario_id"],
        "map_id": row["map_id"],
        "target_id": row["target_id"],
        "layout_id": row["layout_id"],
        "layout_identity_sha256": layout.layout_identity_sha256,
        "route_id": row["route_id"],
        "route_identity_sha256": route.route_identity_sha256,
        "split": row["split"],
        "seed": row["seed"],
        "base_scenario_config_hash": row["base_scenario_config_hash"],
        "applied_visual_randomization_fields": protocol["randomization"]["applied_visual_fields"],
        "randomization_fields_not_applied_to_visuals": protocol["randomization"]["explicitly_not_applied"],
        "generator_version": layout.generator_version,
        "weather": layout.weather,
        "light_intensity": layout.light_intensity,
        "camera_noise_stddev": layout.camera_noise_stddev,
        "camera_pitch_down_deg": protocol["recording"]["camera_pitch_down_deg"],
        "camera_intrinsics_policy": protocol["recording"]["camera_intrinsics_policy"],
        "camera_heading_offset_deg": protocol["recording"]["camera_heading_offset_deg"],
        **timing,
    }
    report["visual_scenario_config_hash"] = object_sha256(report)
    report["world_sha256"] = file_sha256(paths["world"])
    report["planner_config_sha256"] = file_sha256(paths["planner"])
    report["route_manifest_sha256"] = file_sha256(paths["route"])
    report["camera_model_sha256"] = file_sha256(paths["camera"])
    write_json(paths["report"], report)


def _runtime_record(output_root, row, paths, timing):
    recording_directory = Path(output_root) / "recordings" / row["recording_id"]
    flight_events_path = recording_directory / "flight_events.jsonl"
    return {
        "scenario": row,
        "scenario_report": str(paths["report"]),
        "world_path": str(paths["world"]),
        "planner_path": str(paths["planner"]),
        "route_path": str(paths["route"]),
        "recording_directory": str(recording_directory),
        "launcher_environment": {
            "MAP_ID": "custom",
            "WORLD_NAME": f"visual_{row['layout_id']}",
            "WORLD_SRC": str(paths["world"].resolve()),
            "PX4_GZ_MODEL_POSE": "-20,-20,0,0,0,0",
            "SIM_MODEL": "x500_research",
            "RESEARCH_MODEL_SRC": str(paths["camera"].resolve()),
            "HEADLESS": "1",
        },
        "launcher_command": ["bash", "scripts/flight/start_px4_substation.sh"],
        "flight_command": [
            "python", "main.py", "task", "run", "fly_round_trip", "--",
            "--obstacle-config", str(paths["planner"]),
            "--visual-route", str(paths["route"]),
            "--return-home",
            "--visual-mission-events", str(flight_events_path),
        ],
        "flight_events_path": str(flight_events_path),
        **timing,
    }


def prepare_v2_collection_scenario(plan, row, output_root, protocol):
    layout = build_layout_manifest(
        row["layout_id"],
        row["split"],
        row["layout_seed"],
        protocol["randomization"]["configuration"],
    )
    route = build_visual_route(
        layout,
        row["route_id"],
        row["target_class"],
        camera_heading_offset_deg=protocol["recording"]["camera_heading_offset_deg"],
    )
    timing = route_timing(route, protocol["flight_timeout_policy"])
    if layout.layout_identity_sha256 != row["layout_identity_sha256"]:
        raise ValueError("scenario layout does not match collection plan")
    if route.route_identity_sha256 != row["route_identity_sha256"]:
        raise ValueError("scenario route does not match collection plan")
    paths = _scenario_paths(output_root, row)
    _write_scenario_inputs(
        paths,
        layout,
        route,
        protocol["recording"]["camera_pitch_down_deg"],
    )
    _write_scenario_report(paths, plan, row, protocol, layout, route, timing)
    return _runtime_record(output_root, row, paths, timing)
