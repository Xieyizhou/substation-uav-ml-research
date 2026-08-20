"""Runtime Gazebo blocker injection for deterministic replan benchmarks."""

from __future__ import annotations

import asyncio
import json
from math import hypot
from pathlib import Path

from src.flight.flight_state import local_position, publish_mission_event
from src.ml.artifacts import object_sha256


def load_dynamic_scenario(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("dynamic_replanning_scenario_identity_sha256", None)
    if value.get("dynamic_replanning_scenario_schema_version") != 1:
        raise ValueError("unsupported dynamic replanning scenario schema")
    if supplied != object_sha256(value):
        raise ValueError("dynamic replanning scenario identity mismatch")
    value["dynamic_replanning_scenario_identity_sha256"] = supplied
    return value


def blocker_sdf(blocker):
    size = float(blocker["size_m"])
    height = float(blocker["height_m"])
    return (
        "<sdf version='1.9'><model name='"
        + blocker["id"]
        + "'><static>true</static><pose>"
        + f"{blocker['world_east_m']} {blocker['world_north_m']} {height / 2} 0 0 0"
        + "</pose><link name='body'><collision name='collision'><geometry><box><size>"
        + f"{size} {size} {height}"
        + "</size></box></geometry></collision><visual name='visual'><geometry><box><size>"
        + f"{size} {size} {height}"
        + "</size></box></geometry><material><ambient>0.8 0.12 0.08 1</ambient>"
        + "<diffuse>0.8 0.12 0.08 1</diffuse></material></visual></link></model></sdf>"
    )


def gazebo_spawn_command(scenario, gz_command="gz"):
    service = f"/world/{scenario['world_name']}/create"
    request = f"sdf: {json.dumps(blocker_sdf(scenario['blocker']))}"
    return [
        gz_command,
        "service",
        "-s",
        service,
        "--reqtype",
        "gz.msgs.EntityFactory",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "5000",
        "--req",
        request,
    ]


async def spawn_gazebo_blocker(scenario, gz_command="gz"):
    process = await asyncio.create_subprocess_exec(
        *gazebo_spawn_command(scenario, gz_command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    text = (stdout + stderr).decode("utf-8", errors="replace")
    if process.returncode != 0 or "data: true" not in text.lower():
        raise RuntimeError(f"Gazebo rejected dynamic blocker: {text.strip()}")
    return text


def trigger_reached(latest, phase_state, scenario):
    if phase_state.get("route_direction") != scenario["route_direction"]:
        return False
    position = local_position(latest)
    if position is None:
        return False
    resolution = float(scenario.get("resolution_m", 1.0))
    trigger_x, trigger_y = scenario["trigger"]["grid_cell"]
    trigger_east = (float(trigger_x) + 0.5) * resolution
    trigger_north = (float(trigger_y) + 0.5) * resolution
    return hypot(position.east_m - trigger_east, position.north_m - trigger_north) <= float(
        scenario["trigger"]["radius_m"]
    )


async def coordinate_dynamic_blocker(
    latest, phase_state, scenario, *, spawn=spawn_gazebo_blocker, poll_s=0.1
):
    while phase_state.get("phase") not in {"landing", "landed"}:
        if trigger_reached(latest, phase_state, scenario):
            await spawn(scenario)
            publish_mission_event(
                phase_state,
                "dynamic_blocker_spawned",
                blocker_id=scenario["blocker"]["id"],
                grid_cell=scenario["blocker"]["grid_cell"],
                route_direction=scenario["route_direction"],
            )
            phase_state["_dynamic_blocker_spawned"] = True
            return
        await asyncio.sleep(poll_s)
    raise RuntimeError("mission ended before the dynamic blocker was injected")
