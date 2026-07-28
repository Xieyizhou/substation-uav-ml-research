"""Keep the Gazebo red goal marker aligned with the selected A* destination."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from src.maps.map_catalog import project_path, px4_launcher_pid


MARKER_LOCAL_Z_M = 0.035
MARKER_NAMES = ("substation_map::goal_marker", "goal_marker")
PX4_WORLD_SYSTEMS = (
    ("gz-sim-physics-system", "gz::sim::systems::Physics"),
    ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
    ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
    ("gz-sim-contact-system", "gz::sim::systems::Contact"),
    ("gz-sim-imu-system", "gz::sim::systems::Imu"),
    ("gz-sim-air-pressure-system", "gz::sim::systems::AirPressure"),
    ("gz-sim-air-speed-system", "gz::sim::systems::AirSpeed"),
    ("gz-sim-apply-link-wrench-system", "gz::sim::systems::ApplyLinkWrench"),
    ("gz-sim-navsat-system", "gz::sim::systems::NavSat"),
    ("gz-sim-magnetometer-system", "gz::sim::systems::Magnetometer"),
    ("gz-sim-sensors-system", "gz::sim::systems::Sensors"),
)


def _map_metadata(entry):
    config = json.loads(project_path(entry["obstacle_config"]).read_text())
    resolution_m = float(config.get("resolution_m", 1.0))
    origin = [float(value) for value in config.get("gazebo_world_origin_m", [0, 0, 0])]
    return resolution_m, origin


def target_local_position(entry, target):
    """Return the target cell center in map-local Gazebo coordinates."""
    resolution_m, _ = _map_metadata(entry)
    x, y = (int(value) for value in target["cell"])
    return (
        (x + 0.5) * resolution_m,
        (y + 0.5) * resolution_m,
        MARKER_LOCAL_Z_M,
    )


def target_world_position(entry, target):
    """Return the target cell center in Gazebo world coordinates."""
    local_x, local_y, local_z = target_local_position(entry, target)
    _, origin = _map_metadata(entry)
    return local_x + origin[0], local_y + origin[1], local_z + origin[2]


def ensure_px4_world_systems(world):
    """Embed only the standard systems needed by the project's x500 worlds.

    Explicit systems prevent Gazebo from falling back to PX4's global server
    config, which also loads optional optical-flow and GStreamer camera
    plugins. Those plugins are not used by this project and are sensitive to
    Homebrew multimedia dependency upgrades.
    """
    existing_names = {
        plugin.get("name")
        for plugin in world.findall("plugin")
        if plugin.get("name")
    }
    for filename, name in PX4_WORLD_SYSTEMS:
        if name in existing_names:
            continue
        plugin = ET.SubElement(world, "plugin", filename=filename, name=name)
        if name == "gz::sim::systems::Sensors":
            render_engine = ET.SubElement(plugin, "render_engine")
            render_engine.text = "ogre2"


def prepare_world_with_target(source_path, output_path, entry, target):
    """Write a runtime SDF copy whose red goal marker uses the selected target."""
    tree = ET.parse(source_path)
    world = tree.getroot().find("world")
    if world is None:
        raise ValueError(f"Expected one world element in {source_path}")
    ensure_px4_world_systems(world)
    markers = [
        model
        for model in tree.getroot().iter("model")
        if model.get("name") == "goal_marker"
    ]
    if len(markers) != 1:
        raise ValueError(
            f"Expected one goal_marker in {source_path}, found {len(markers)}"
        )

    pose = markers[0].find("pose")
    if pose is None:
        pose = ET.SubElement(markers[0], "pose")
    values = (pose.text or "0 0 0.035 0 0 0").split()
    values += ["0"] * (6 - len(values))
    local_x, local_y, local_z = target_local_position(entry, target)
    values[:3] = [f"{local_x:g}", f"{local_y:g}", f"{local_z:g}"]
    pose.text = " ".join(values[:6])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _set_pose_request(name, position):
    x, y, z = position
    return (
        f'name: "{name}" '
        f"position {{ x: {x:g} y: {y:g} z: {z:g} }} "
        "orientation { w: 1 }"
    )


def sync_running_goal_marker(entry, target, timeout_s=2.0):
    """Move the current Gazebo marker; return `(status, detail)` without raising."""
    if px4_launcher_pid() is None:
        return "not_running", "PX4/Gazebo is not running"
    gz_command = shutil.which("gz")
    if gz_command is None:
        return "unavailable", "the `gz` command was not found"

    service = f"/world/{entry['world_name']}/set_pose"
    position = target_world_position(entry, target)
    details = []
    for marker_name in MARKER_NAMES:
        command = [
            gz_command,
            "service",
            "-s",
            service,
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(int(timeout_s * 1000)),
            "--req",
            _set_pose_request(marker_name, position),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s + 1.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            details.append(str(error))
            continue
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode == 0 and "true" in output.lower():
            return "updated", marker_name
        details.append(output or f"exit code {result.returncode}")
    return "failed", "; ".join(details)
