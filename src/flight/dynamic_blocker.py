"""Runtime Gazebo blocker injection for deterministic replan benchmarks."""

from __future__ import annotations

import asyncio
import json
from math import hypot
import os
from pathlib import Path
import shutil
import subprocess

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


GAZEBO_SERVICE_TIMEOUT_MS = 20_000
GAZEBO_SPAWN_ATTEMPTS = 3


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
        str(GAZEBO_SERVICE_TIMEOUT_MS),
        "--req",
        request,
    ]


def _run_command(command, timeout_s):
    """Run Gazebo with macOS posix_spawn after MAVSDK starts gRPC threads."""
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(f"Gazebo command not found: {command[0]}")
    resolved = [executable, *command[1:]]
    environment = os.environ.copy()
    environment.setdefault("GZ_IP", "127.0.0.1")
    return subprocess.run(
        resolved,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        # CPython's macOS fast path requires an absolute executable and
        # close_fds=False. This avoids fork() after MAVSDK has started gRPC
        # worker threads, which otherwise corrupts Gazebo transport polling.
        close_fds=False,
        env=environment,
        check=False,
    )


async def _gazebo_model_exists(model_id, gz_command):
    try:
        result = await asyncio.to_thread(
            _run_command, [gz_command, "model", "--list"], 15.0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return result.returncode == 0 and model_id in names


def _timeout_output(error):
    parts = []
    for value in (error.stdout, error.stderr):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value:
            parts.append(str(value))
    return "".join(parts)


async def spawn_gazebo_blocker(scenario, gz_command="gz"):
    command = gazebo_spawn_command(scenario, gz_command)
    model_id = scenario["blocker"]["id"]
    failures = []
    for attempt in range(1, GAZEBO_SPAWN_ATTEMPTS + 1):
        try:
            result = await asyncio.to_thread(
                _run_command,
                command,
                GAZEBO_SERVICE_TIMEOUT_MS / 1000.0 + 5.0,
            )
            text = result.stdout
        except subprocess.TimeoutExpired as error:
            text = _timeout_output(error)
            result = None
        if result is not None and result.returncode == 0 and "data: true" in text.lower():
            return text
        if await _gazebo_model_exists(model_id, gz_command):
            return text
        failures.append(f"attempt {attempt}: {text.strip() or 'no response'}")
        if attempt < GAZEBO_SPAWN_ATTEMPTS:
            await asyncio.sleep(0.5)
    raise RuntimeError("Gazebo rejected dynamic blocker: " + " | ".join(failures))


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
