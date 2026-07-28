"""Deterministic simulator domain-randomization manifests."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
import xml.etree.ElementTree as ET

from src.ml.artifacts import object_sha256, write_json


SCENE_RANDOMIZATION_VERSION = "gazebo-sdf-v2"
MAP_SIZE_M = {
    "training": 16.0,
    "simple": 20.0,
    "medium": 24.0,
    "complex": 28.0,
    "extreme": 32.0,
}


def load_ranges(path: Path):
    with Path(path).open(encoding="utf-8") as source:
        values = json.load(source)
    if values.get("schema_version") != 1:
        raise ValueError("unsupported domain-randomization schema")
    return values


def sample_manifest(config, *, map_id, seed):
    generator = random.Random(int(seed))

    def sample_pair(name):
        lower, upper = config[name]
        return generator.uniform(float(lower), float(upper))

    count_min, count_max = config.get("unknown_obstacle_count", [0, 0])
    obstacle_count = generator.randint(int(count_min), int(count_max))
    size_min, size_max = config.get("unknown_obstacle_size_m", [0.4, 1.0])
    map_size = MAP_SIZE_M.get(str(map_id), 20.0)
    obstacles = [
        {
            "id": f"unknown_{index + 1:02d}",
            "east_m": generator.uniform(3.0, map_size - 3.0),
            "north_m": generator.uniform(3.0, map_size - 3.0),
            "size_m": generator.uniform(float(size_min), float(size_max)),
            "height_m": generator.uniform(0.6, 2.4),
        }
        for index in range(obstacle_count)
    ]
    manifest = {
        "schema_version": 1,
        "generator_version": SCENE_RANDOMIZATION_VERSION,
        "map_id": str(map_id),
        "seed": int(seed),
        "equipment_scale": sample_pair("equipment_scale"),
        "equipment_position_jitter_m": sample_pair("equipment_position_jitter_m"),
        "light_intensity": sample_pair("light_intensity"),
        "material_age": sample_pair("material_age"),
        "lidar_noise_stddev_m": sample_pair("lidar_noise_stddev_m"),
        "lidar_dropout_probability": sample_pair("lidar_dropout_probability"),
        "attitude_jitter_deg": sample_pair("attitude_jitter_deg"),
        "camera_noise_stddev": sample_pair("camera_noise_stddev"),
        "sensor_outage_probability": sample_pair("sensor_outage_probability"),
        "sensor_outage_duration_s": sample_pair("sensor_outage_duration_s"),
        "unknown_obstacles": obstacles,
        "weather": generator.choice(list(config["weather"])),
    }
    manifest["config_hash"] = object_sha256(manifest)
    return manifest


_NON_EQUIPMENT = {
    "ground_plane",
    "substation_floor",
    "substation_floor_grid",
    "map_axes",
    "map_origin_marker",
    "east_axis_marker",
    "north_axis_marker",
    "service_corridor_1",
    "service_corridor_2",
    "service_corridor_east_west",
    "service_corridor_north_south",
    "start_marker",
    "goal_marker",
}


def _number_list(element):
    return [float(value) for value in (element.text or "").split()]


def _set_numbers(element, values):
    element.text = " ".join(f"{value:.6g}" for value in values)


def _randomize_equipment(parent, manifest):
    generator = random.Random(int(manifest["seed"]) ^ 0x5F3759DF)
    changes = []
    for model in parent.findall("./model"):
        name = model.get("name", "")
        if name in _NON_EQUIPMENT or name.startswith("fence"):
            continue
        pose = model.find("./pose")
        if pose is None:
            continue
        values = _number_list(pose)
        if len(values) != 6:
            continue
        jitter = float(manifest["equipment_position_jitter_m"])
        offset_east = generator.uniform(-abs(jitter), abs(jitter))
        offset_north = generator.uniform(-abs(jitter), abs(jitter))
        values[0] += offset_east
        values[1] += offset_north
        _set_numbers(pose, values)
        scale = generator.uniform(
            min(1.0, float(manifest["equipment_scale"])),
            max(1.0, float(manifest["equipment_scale"])),
        )
        for size in model.findall(".//geometry/box/size"):
            dimensions = _number_list(size)
            if len(dimensions) == 3:
                _set_numbers(size, [value * scale for value in dimensions])
        for radius in model.findall(".//geometry/cylinder/radius"):
            radius.text = f"{float(radius.text) * scale:.6g}"
        changes.append(
            {
                "model": name,
                "east_offset_m": offset_east,
                "north_offset_m": offset_north,
                "scale": scale,
            }
        )
    return changes


def _unknown_model(specification):
    model = ET.Element("model", name=specification["id"])
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = (
        f"{specification['east_m']} {specification['north_m']} 0 0 0 0"
    )
    link = ET.SubElement(model, "link", name="link")
    for kind in ("collision", "visual"):
        item = ET.SubElement(link, kind, name=kind)
        ET.SubElement(item, "pose").text = (
            f"0 0 {specification['height_m'] / 2:.6g} 0 0 0"
        )
        geometry = ET.SubElement(item, "geometry")
        box = ET.SubElement(geometry, "box")
        ET.SubElement(box, "size").text = (
            f"{specification['size_m']} {specification['size_m']} "
            f"{specification['height_m']}"
        )
        if kind == "visual":
            material = ET.SubElement(item, "material")
            ET.SubElement(material, "ambient").text = "0.7 0.15 0.1 1"
            ET.SubElement(material, "diffuse").text = "0.8 0.2 0.1 1"
    return model


def materialize_world(source_path, output_path, manifest, *, report_path=None):
    """Create a launchable randomized SDF without modifying source worlds."""
    tree = ET.parse(source_path)
    root = tree.getroot()
    world = root.find("./world")
    if world is None:
        raise ValueError("SDF does not contain a world")
    station = world.find("./model[@name='substation_map']")
    equipment_parent = station if station is not None else world
    changes = _randomize_equipment(equipment_parent, manifest)
    for specification in manifest.get("unknown_obstacles", []):
        equipment_parent.append(_unknown_model(specification))
    sun = world.find("./light[@name='sun']/diffuse")
    if sun is not None:
        intensity = float(manifest["light_intensity"])
        _set_numbers(sun, [0.8 * intensity] * 3 + [1.0])
    ET.indent(tree, space="  ")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    report = {
        **manifest,
        "source_world": str(source_path),
        "generated_world": str(output_path),
        "scenario_config_hash": manifest["config_hash"],
        "equipment_changes": changes,
        "unknown_obstacles": manifest.get("unknown_obstacles", []),
        "sensor_faults": {
            key: manifest[key]
            for key in (
                "lidar_noise_stddev_m",
                "lidar_dropout_probability",
                "sensor_outage_probability",
                "sensor_outage_duration_s",
                "attitude_jitter_deg",
            )
        },
    }
    if report_path:
        write_json(report_path, report)
    return report


def materialize_planner_config(source_path, output_path, report):
    """Create the oracle grid that matches a randomized simulator world."""
    with Path(source_path).open(encoding="utf-8") as source:
        config = json.load(source)
    resolution = float(config.get("resolution_m", 1.0))
    width, height = int(config["width"]), int(config["height"])
    changes = {item["model"]: item for item in report.get("equipment_changes", [])}

    def clamp(value, maximum):
        return min(max(int(value), 0), maximum - 1)

    randomized = []
    for obstacle in config.get("obstacles", []):
        obstacle = dict(obstacle)
        change = changes.get(obstacle.get("name"))
        if change:
            dx = round(float(change["east_offset_m"]) / resolution)
            dy = round(float(change["north_offset_m"]) / resolution)
            if obstacle.get("type") == "cell":
                obstacle["x"] = clamp(obstacle["x"] + dx, width)
                obstacle["y"] = clamp(obstacle["y"] + dy, height)
            elif obstacle.get("type") == "rect":
                center_x = (obstacle["x_min"] + obstacle["x_max"]) / 2 + dx
                center_y = (obstacle["y_min"] + obstacle["y_max"]) / 2 + dy
                half_x = (
                    (obstacle["x_max"] - obstacle["x_min"] + 1)
                    * float(change["scale"])
                    / 2
                )
                half_y = (
                    (obstacle["y_max"] - obstacle["y_min"] + 1)
                    * float(change["scale"])
                    / 2
                )
                obstacle.update(
                    {
                        "x_min": clamp(math.floor(center_x - half_x + 0.5), width),
                        "x_max": clamp(math.ceil(center_x + half_x - 0.5), width),
                        "y_min": clamp(math.floor(center_y - half_y + 0.5), height),
                        "y_max": clamp(math.ceil(center_y + half_y - 0.5), height),
                    }
                )
        randomized.append(obstacle)
    for item in report.get("unknown_obstacles", []):
        half = float(item["size_m"]) / 2
        randomized.append(
            {
                "name": item["id"],
                "type": "rect",
                "x_min": clamp(math.floor((item["east_m"] - half) / resolution), width),
                "x_max": clamp(math.floor((item["east_m"] + half) / resolution), width),
                "y_min": clamp(math.floor((item["north_m"] - half) / resolution), height),
                "y_max": clamp(math.floor((item["north_m"] + half) / resolution), height),
                "z_min_m": 0.0,
                "z_max_m": float(item["height_m"]),
                "visual_category": "unknown_obstacle",
            }
        )
    config["map_name"] = f"{config.get('map_name', 'map')}_randomized_{report['seed']}"
    config["scenario_config_hash"] = report["scenario_config_hash"]
    config["obstacles"] = randomized
    write_json(output_path, config)
    return config
