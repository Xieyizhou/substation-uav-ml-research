#!/usr/bin/env python3
"""Generate aligned Gazebo worlds and A* configs for test substations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


MAP_SPECS = [
    {
        "id": "training",
        "world_name": "substation_training",
        "map_name": "substation_training_v1",
        "display_name": "Training Yard",
        "difficulty": 1,
        "difficulty_label": "简单",
        "description": "16×16 m 入门场地，障碍少、通道宽，适合首次飞行和回归测试。",
        "width": 16,
        "height": 16,
        "start_cell": [0, 0],
        "goal_cell": [13, 13],
        "corridors": [
            {"x": 8.0, "y": 2.5, "size_x": 12.0, "size_y": 0.5},
            {"x": 2.5, "y": 8.0, "size_x": 0.5, "size_y": 12.0},
        ],
        "obstacles": [
            {"name": "entry_switchgear", "type": "rect", "x_min": 5, "x_max": 6, "y_min": 0, "y_max": 2, "z_min_m": 0.0, "z_max_m": 1.4, "visual_category": "switchgear"},
            {"name": "transformer_a", "type": "rect", "x_min": 5, "x_max": 7, "y_min": 5, "y_max": 7, "z_min_m": 0.0, "z_max_m": 1.8, "visual_category": "transformer"},
            {"name": "cabinet_north", "type": "rect", "x_min": 3, "x_max": 4, "y_min": 10, "y_max": 11, "z_min_m": 0.0, "z_max_m": 1.4, "visual_category": "cabinet"},
            {"name": "switchgear_east", "type": "rect", "x_min": 10, "x_max": 12, "y_min": 3, "y_max": 4, "z_min_m": 0.0, "z_max_m": 1.4, "visual_category": "switchgear"},
            {"name": "pole_center", "type": "cell", "x": 10, "y": 10, "z_min_m": 0.0, "z_max_m": 4.2, "visual_category": "pole"},
        ],
    },
    {
        "id": "medium",
        "world_name": "substation_medium",
        "map_name": "substation_medium_v1",
        "display_name": "Dual-Bay Station",
        "difficulty": 3,
        "difficulty_label": "中等",
        "description": "24×24 m 双间隔变电站，包含多种设备和交叉服务通道。",
        "width": 24,
        "height": 24,
        "start_cell": [0, 0],
        "goal_cell": [21, 21],
        "corridors": [
            {"x": 12.0, "y": 3.5, "size_x": 18.0, "size_y": 0.55},
            {"x": 3.5, "y": 12.0, "size_x": 0.55, "size_y": 18.0},
            {"x": 14.0, "y": 19.5, "size_x": 14.0, "size_y": 0.55},
        ],
        "obstacles": [
            {"name": "entry_switchgear", "type": "rect", "x_min": 5, "x_max": 6, "y_min": 0, "y_max": 2, "z_min_m": 0.0, "z_max_m": 1.4, "visual_category": "switchgear"},
            {"name": "transformer_west", "type": "rect", "x_min": 6, "x_max": 8, "y_min": 5, "y_max": 7, "z_min_m": 0.0, "z_max_m": 1.9, "visual_category": "transformer"},
            {"name": "transformer_north", "type": "rect", "x_min": 15, "x_max": 17, "y_min": 14, "y_max": 16, "z_min_m": 0.0, "z_max_m": 1.9, "visual_category": "transformer"},
            {"name": "cabinet_west", "type": "rect", "x_min": 4, "x_max": 5, "y_min": 15, "y_max": 17, "z_min_m": 0.0, "z_max_m": 1.5, "visual_category": "cabinet"},
            {"name": "switchgear_south", "type": "rect", "x_min": 12, "x_max": 14, "y_min": 3, "y_max": 4, "z_min_m": 0.0, "z_max_m": 1.4, "visual_category": "switchgear"},
            {"name": "capacitor_east", "type": "rect", "x_min": 18, "x_max": 20, "y_min": 7, "y_max": 9, "z_min_m": 0.0, "z_max_m": 1.7, "visual_category": "capacitor_bank"},
            {"name": "pole_a", "type": "cell", "x": 10, "y": 11, "z_min_m": 0.0, "z_max_m": 4.5, "visual_category": "pole"},
            {"name": "pole_b", "type": "cell", "x": 12, "y": 18, "z_min_m": 0.0, "z_max_m": 4.5, "visual_category": "pole"},
            {"name": "pole_c", "type": "cell", "x": 20, "y": 15, "z_min_m": 0.0, "z_max_m": 4.5, "visual_category": "pole"},
        ],
    },
    {
        "id": "complex",
        "world_name": "substation_complex",
        "map_name": "substation_complex_v1",
        "display_name": "Multi-Bay Station",
        "difficulty": 4,
        "difficulty_label": "复杂",
        "description": "28×28 m 多间隔场地，设备类型丰富、通道更窄，适合重规划测试。",
        "width": 28,
        "height": 28,
        "start_cell": [0, 0],
        "goal_cell": [25, 25],
        "corridors": [
            {"x": 14.0, "y": 3.5, "size_x": 22.0, "size_y": 0.55},
            {"x": 3.5, "y": 14.0, "size_x": 0.55, "size_y": 22.0},
            {"x": 15.0, "y": 13.5, "size_x": 20.0, "size_y": 0.55},
            {"x": 14.5, "y": 23.5, "size_x": 17.0, "size_y": 0.55},
        ],
        "obstacles": [
            {"name": "entry_switchgear", "type": "rect", "x_min": 5, "x_max": 6, "y_min": 0, "y_max": 2, "z_min_m": 0.0, "z_max_m": 1.4, "visual_category": "switchgear"},
            {"name": "west_switchgear_01", "type": "rect", "x_min": 0, "x_max": 1, "y_min": 5, "y_max": 7, "z_min_m": 0.0, "z_max_m": 1.6, "visual_category": "switchgear"},
            {"name": "west_switchgear_02", "type": "rect", "x_min": 0, "x_max": 1, "y_min": 9, "y_max": 11, "z_min_m": 0.0, "z_max_m": 1.6, "visual_category": "switchgear"},
            {"name": "west_switchgear_03", "type": "rect", "x_min": 0, "x_max": 1, "y_min": 13, "y_max": 15, "z_min_m": 0.0, "z_max_m": 1.6, "visual_category": "switchgear"},
            {"name": "west_switchgear_04", "type": "rect", "x_min": 0, "x_max": 1, "y_min": 17, "y_max": 18, "z_min_m": 0.0, "z_max_m": 1.6, "visual_category": "switchgear"},
            {"name": "transformer_sw", "type": "rect", "x_min": 5, "x_max": 8, "y_min": 5, "y_max": 8, "z_min_m": 0.0, "z_max_m": 2.0, "visual_category": "transformer"},
            {"name": "transformer_se", "type": "rect", "x_min": 15, "x_max": 18, "y_min": 6, "y_max": 9, "z_min_m": 0.0, "z_max_m": 2.0, "visual_category": "transformer"},
            {"name": "transformer_mid", "type": "rect", "x_min": 10, "x_max": 13, "y_min": 16, "y_max": 19, "z_min_m": 0.0, "z_max_m": 2.0, "visual_category": "transformer"},
            {"name": "control_building", "type": "rect", "x_min": 20, "x_max": 23, "y_min": 18, "y_max": 21, "z_min_m": 0.0, "z_max_m": 2.8, "visual_category": "control_building"},
            {"name": "switchgear_west", "type": "rect", "x_min": 3, "x_max": 5, "y_min": 15, "y_max": 17, "z_min_m": 0.0, "z_max_m": 1.5, "visual_category": "switchgear"},
            {"name": "capacitor_east", "type": "rect", "x_min": 20, "x_max": 22, "y_min": 11, "y_max": 13, "z_min_m": 0.0, "z_max_m": 1.8, "visual_category": "capacitor_bank"},
            {"name": "reactor_north", "type": "rect", "x_min": 6, "x_max": 8, "y_min": 22, "y_max": 24, "z_min_m": 0.0, "z_max_m": 2.2, "visual_category": "reactor"},
            {"name": "cabinet_center", "type": "rect", "x_min": 16, "x_max": 17, "y_min": 14, "y_max": 15, "z_min_m": 0.0, "z_max_m": 1.5, "visual_category": "cabinet"},
            {"name": "pole_a", "type": "cell", "x": 11, "y": 11, "z_min_m": 0.0, "z_max_m": 4.8, "visual_category": "pole"},
            {"name": "pole_b", "type": "cell", "x": 18, "y": 16, "z_min_m": 0.0, "z_max_m": 4.8, "visual_category": "pole"},
            {"name": "pole_c", "type": "cell", "x": 24, "y": 10, "z_min_m": 0.0, "z_max_m": 4.8, "visual_category": "pole"},
            {"name": "pole_d", "type": "cell", "x": 14, "y": 23, "z_min_m": 0.0, "z_max_m": 4.8, "visual_category": "pole"},
        ],
    },
    {
        "id": "extreme",
        "world_name": "substation_extreme",
        "map_name": "substation_extreme_v1",
        "display_name": "Dense Multi-Voltage Station",
        "difficulty": 5,
        "difficulty_label": "很复杂",
        "description": "32×32 m 高密度多电压等级场地，用于长路径、窄通道和主动重规划压力测试。",
        "width": 32,
        "height": 32,
        "start_cell": [0, 0],
        "goal_cell": [29, 29],
        "corridors": [
            {"x": 16.0, "y": 3.5, "size_x": 26.0, "size_y": 0.55},
            {"x": 3.5, "y": 16.0, "size_x": 0.55, "size_y": 26.0},
            {"x": 17.0, "y": 14.0, "size_x": 24.0, "size_y": 0.55},
            {"x": 15.5, "y": 24.0, "size_x": 21.0, "size_y": 0.55},
            {"x": 12.0, "y": 18.0, "size_x": 0.55, "size_y": 20.0},
        ],
        "obstacles": [
            {"name": "entry_switchgear", "type": "rect", "x_min": 5, "x_max": 6, "y_min": 0, "y_max": 2, "z_min_m": 0.0, "z_max_m": 1.4, "visual_category": "switchgear"},
            {"name": "transformer_01", "type": "rect", "x_min": 5, "x_max": 8, "y_min": 5, "y_max": 8, "z_min_m": 0.0, "z_max_m": 2.1, "visual_category": "transformer"},
            {"name": "transformer_02", "type": "rect", "x_min": 14, "x_max": 17, "y_min": 4, "y_max": 7, "z_min_m": 0.0, "z_max_m": 2.1, "visual_category": "transformer"},
            {"name": "transformer_03", "type": "rect", "x_min": 22, "x_max": 25, "y_min": 6, "y_max": 9, "z_min_m": 0.0, "z_max_m": 2.1, "visual_category": "transformer"},
            {"name": "transformer_04", "type": "rect", "x_min": 7, "x_max": 10, "y_min": 17, "y_max": 20, "z_min_m": 0.0, "z_max_m": 2.1, "visual_category": "transformer"},
            {"name": "transformer_05", "type": "rect", "x_min": 19, "x_max": 22, "y_min": 18, "y_max": 21, "z_min_m": 0.0, "z_max_m": 2.1, "visual_category": "transformer"},
            {"name": "control_building", "type": "rect", "x_min": 25, "x_max": 28, "y_min": 23, "y_max": 26, "z_min_m": 0.0, "z_max_m": 3.0, "visual_category": "control_building"},
            {"name": "switchgear_south", "type": "rect", "x_min": 9, "x_max": 11, "y_min": 10, "y_max": 12, "z_min_m": 0.0, "z_max_m": 1.6, "visual_category": "switchgear"},
            {"name": "switchgear_center", "type": "rect", "x_min": 15, "x_max": 17, "y_min": 13, "y_max": 15, "z_min_m": 0.0, "z_max_m": 1.6, "visual_category": "switchgear"},
            {"name": "switchgear_north", "type": "rect", "x_min": 13, "x_max": 15, "y_min": 24, "y_max": 26, "z_min_m": 0.0, "z_max_m": 1.6, "visual_category": "switchgear"},
            {"name": "capacitor_01", "type": "rect", "x_min": 25, "x_max": 27, "y_min": 13, "y_max": 15, "z_min_m": 0.0, "z_max_m": 1.8, "visual_category": "capacitor_bank"},
            {"name": "capacitor_02", "type": "rect", "x_min": 4, "x_max": 6, "y_min": 25, "y_max": 27, "z_min_m": 0.0, "z_max_m": 1.8, "visual_category": "capacitor_bank"},
            {"name": "reactor_01", "type": "rect", "x_min": 16, "x_max": 18, "y_min": 27, "y_max": 29, "z_min_m": 0.0, "z_max_m": 2.3, "visual_category": "reactor"},
            {"name": "reactor_02", "type": "rect", "x_min": 26, "x_max": 28, "y_min": 4, "y_max": 5, "z_min_m": 0.0, "z_max_m": 2.3, "visual_category": "reactor"},
            {"name": "cabinet_west", "type": "rect", "x_min": 4, "x_max": 5, "y_min": 12, "y_max": 14, "z_min_m": 0.0, "z_max_m": 1.5, "visual_category": "cabinet"},
            {"name": "cabinet_north", "type": "rect", "x_min": 20, "x_max": 21, "y_min": 27, "y_max": 28, "z_min_m": 0.0, "z_max_m": 1.5, "visual_category": "cabinet"},
            {"name": "pole_a", "type": "cell", "x": 12, "y": 9, "z_min_m": 0.0, "z_max_m": 5.0, "visual_category": "pole"},
            {"name": "pole_b", "type": "cell", "x": 20, "y": 11, "z_min_m": 0.0, "z_max_m": 5.0, "visual_category": "pole"},
            {"name": "pole_c", "type": "cell", "x": 12, "y": 21, "z_min_m": 0.0, "z_max_m": 5.0, "visual_category": "pole"},
            {"name": "pole_d", "type": "cell", "x": 23, "y": 25, "z_min_m": 0.0, "z_max_m": 5.0, "visual_category": "pole"},
            {"name": "pole_e", "type": "cell", "x": 29, "y": 18, "z_min_m": 0.0, "z_max_m": 5.0, "visual_category": "pole"},
        ],
    },
]


CATEGORY_COLORS = {
    "transformer": "0.07 0.16 0.19 1",
    "cabinet": "0.02 0.32 0.43 1",
    "switchgear": "0.08 0.38 0.48 1",
    "capacitor_bank": "0.16 0.35 0.40 1",
    "reactor": "0.24 0.29 0.34 1",
    "control_building": "0.32 0.34 0.35 1",
    "pole": "0.08 0.09 0.09 1",
}


def text(parent, tag, value, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    element.text = str(value)
    return element


def pose_text(values):
    return " ".join(f"{float(value):g}" for value in values)


def add_material(visual, color):
    material = ET.SubElement(visual, "material")
    text(material, "ambient", color)
    text(material, "diffuse", color)
    text(material, "specular", "0.08 0.08 0.08 1")


def add_box(parent, size):
    geometry = ET.SubElement(parent, "geometry")
    box = ET.SubElement(geometry, "box")
    text(box, "size", pose_text(size))


def add_cylinder(parent, radius, length):
    geometry = ET.SubElement(parent, "geometry")
    cylinder = ET.SubElement(geometry, "cylinder")
    text(cylinder, "radius", f"{radius:g}")
    text(cylinder, "length", f"{length:g}")


def add_box_visual(link, name, pose, size, color):
    visual = ET.SubElement(link, "visual", name=name)
    text(visual, "pose", pose_text(pose))
    add_box(visual, size)
    add_material(visual, color)
    return visual


def add_cylinder_visual(link, name, pose, radius, length, color):
    visual = ET.SubElement(link, "visual", name=name)
    text(visual, "pose", pose_text(pose))
    add_cylinder(visual, radius, length)
    add_material(visual, color)
    return visual


def add_box_collision(link, name, pose, size):
    collision = ET.SubElement(link, "collision", name=name)
    text(collision, "pose", pose_text(pose))
    add_box(collision, size)


def add_cylinder_collision(link, name, pose, radius, length):
    collision = ET.SubElement(link, "collision", name=name)
    text(collision, "pose", pose_text(pose))
    add_cylinder(collision, radius, length)


def add_static_model(parent, name, pose):
    model = ET.SubElement(parent, "model", name=name)
    text(model, "static", "true")
    text(model, "pose", pose_text(pose))
    return model


def obstacle_bounds(obstacle):
    if obstacle["type"] == "cell":
        x_min = x_max = int(obstacle["x"])
        y_min = y_max = int(obstacle["y"])
    else:
        x_min = int(obstacle["x_min"])
        x_max = int(obstacle["x_max"])
        y_min = int(obstacle["y_min"])
        y_max = int(obstacle["y_max"])
    return x_min, x_max, y_min, y_max


def add_equipment(parent, obstacle):
    x_min, x_max, y_min, y_max = obstacle_bounds(obstacle)
    size_x = x_max - x_min + 1
    size_y = y_max - y_min + 1
    center_x = (x_min + x_max + 1) / 2
    center_y = (y_min + y_max + 1) / 2
    z_min = float(obstacle["z_min_m"])
    z_max = float(obstacle["z_max_m"])
    height = z_max - z_min
    center_z = z_min + height / 2
    category = obstacle.get("visual_category", "cabinet")
    color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["cabinet"])

    model = add_static_model(parent, obstacle["name"], [center_x, center_y, 0, 0, 0, 0])
    link = ET.SubElement(model, "link", name="link")

    if category == "pole":
        add_cylinder_collision(link, "collision", [0, 0, center_z, 0, 0, 0], 0.22, height)
        add_cylinder_visual(link, "pole", [0, 0, center_z, 0, 0, 0], 0.18, height, color)
        add_box_visual(link, "crossarm", [0, 0, z_max - 0.2, 0, 0, 0], [1.25, 0.12, 0.12], "0.18 0.16 0.10 1")
        return

    if category == "reactor":
        radius = max(0.35, min(size_x, size_y) * 0.34)
        add_cylinder_collision(link, "collision", [0, 0, center_z, 0, 0, 0], radius, height)
        add_cylinder_visual(link, "reactor", [0, 0, center_z, 0, 0, 0], radius * 0.92, height * 0.92, color)
        add_box_visual(link, "base", [0, 0, 0.08, 0, 0, 0], [size_x * 0.9, size_y * 0.9, 0.16], "0.12 0.12 0.12 1")
        return

    add_box_collision(link, "collision", [0, 0, center_z, 0, 0, 0], [size_x * 0.92, size_y * 0.92, height])
    add_box_visual(link, "base", [0, 0, 0.08, 0, 0, 0], [size_x * 0.96, size_y * 0.96, 0.16], "0.10 0.10 0.10 1")
    add_box_visual(link, "body", [0, 0, center_z, 0, 0, 0], [size_x * 0.86, size_y * 0.86, height * 0.9], color)

    if category == "transformer":
        for index, offset in enumerate((-0.28, 0.0, 0.28), start=1):
            add_cylinder_visual(
                link,
                f"bushing_{index}",
                [offset * size_x, 0, z_max + 0.18, 0, 0, 0],
                0.09,
                0.36,
                "0.75 0.72 0.64 1",
            )
    elif category == "capacitor_bank":
        for row in (-0.22, 0.22):
            for column in (-0.25, 0.0, 0.25):
                add_cylinder_visual(
                    link,
                    f"capacitor_{row}_{column}",
                    [column * size_x, row * size_y, center_z + 0.08, 0, 0, 0],
                    0.1,
                    height * 0.65,
                    "0.72 0.74 0.70 1",
                )
    else:
        add_box_visual(
            link,
            "front_panel",
            [0, -size_y * 0.435, center_z, 0, 0, 0],
            [size_x * 0.62, 0.025, height * 0.58],
            "0.16 0.20 0.21 1",
        )


def add_boundary_fence(parent, width, height):
    color = "0.16 0.16 0.16 1"
    segments = [
        ("fence_north", [width / 2, height - 0.12, 0.45, 0, 0, 0], [width, 0.18, 0.9]),
        ("fence_east", [width - 0.12, height / 2, 0.45, 0, 0, 0], [0.18, height, 0.9]),
        ("fence_south", [(width + 4) / 2, 0.12, 0.45, 0, 0, 0], [width - 4, 0.18, 0.9]),
        ("fence_west", [0.12, (height + 4) / 2, 0.45, 0, 0, 0], [0.18, height - 4, 0.9]),
    ]
    for name, pose, size in segments:
        model = add_static_model(parent, name, pose)
        link = ET.SubElement(model, "link", name="link")
        add_box_collision(link, "collision", [0, 0, 0, 0, 0, 0], size)
        add_box_visual(link, "visual", [0, 0, 0, 0, 0, 0], size, color)


def build_world(spec):
    width = spec["width"]
    height = spec["height"]
    origin = [-width / 2, -height / 2, 0]

    sdf = ET.Element("sdf", version="1.9")
    world = ET.SubElement(sdf, "world", name=spec["world_name"])
    text(world, "gravity", "0 0 -9.81")
    text(world, "magnetic_field", "6e-06 2.3e-05 -4.2e-05")
    ET.SubElement(world, "atmosphere", type="adiabatic")
    scene = ET.SubElement(world, "scene")
    text(scene, "ambient", "0.72 0.72 0.72 1")
    text(scene, "background", "0.72 0.75 0.78 1")

    light = ET.SubElement(world, "light", name="sun", type="directional")
    text(light, "cast_shadows", "true")
    text(light, "pose", "0 0 20 0 0 0")
    text(light, "diffuse", "0.85 0.85 0.82 1")
    text(light, "specular", "0.2 0.2 0.2 1")
    text(light, "direction", "-0.5 0.1 -0.9")

    spherical = ET.SubElement(world, "spherical_coordinates")
    text(spherical, "surface_model", "EARTH_WGS84")
    text(spherical, "world_frame_orientation", "ENU")
    text(spherical, "latitude_deg", "47.397971057728974")
    text(spherical, "longitude_deg", "8.546163739800146")
    text(spherical, "elevation", "0")

    map_model = add_static_model(world, "substation_map", [*origin, 0, 0, 0])

    ground = add_static_model(map_model, "ground_plane", [width / 2, height / 2, 0, 0, 0, 0])
    ground_link = ET.SubElement(ground, "link", name="link")
    collision = ET.SubElement(ground_link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    plane = ET.SubElement(geometry, "plane")
    text(plane, "normal", "0 0 1")
    text(plane, "size", f"{width + 4:g} {height + 4:g}")

    floor = add_static_model(map_model, "substation_floor", [width / 2, height / 2, 0.01, 0, 0, 0])
    floor_link = ET.SubElement(floor, "link", name="link")
    add_box_visual(floor_link, "visual", [0, 0, 0, 0, 0, 0], [width, height, 0.02], "0.42 0.43 0.40 1")

    grid = add_static_model(map_model, "substation_floor_grid", [0, 0, 0, 0, 0, 0])
    grid_link = ET.SubElement(grid, "link", name="link")
    for x in range(width + 1):
        add_box_visual(grid_link, f"x_grid_{x:02d}", [x, height / 2, 0.024, 0, 0, 0], [0.025, height, 0.006], "0.16 0.17 0.16 1")
    for y in range(height + 1):
        add_box_visual(grid_link, f"y_grid_{y:02d}", [width / 2, y, 0.024, 0, 0, 0], [width, 0.025, 0.006], "0.16 0.17 0.16 1")

    axes = add_static_model(map_model, "map_axes", [0, 0, 0, 0, 0, 0])
    axes_link = ET.SubElement(axes, "link", name="link")
    add_box_visual(axes_link, "east_axis", [width / 2, 0, 0.028, 0, 0, 0], [width, 0.045, 0.012], "0.08 0.24 0.95 1")
    add_box_visual(axes_link, "north_axis", [0, height / 2, 0.028, 0, 0, 0], [0.045, height, 0.012], "0.05 0.76 0.18 1")

    for index, corridor in enumerate(spec.get("corridors", []), start=1):
        model = add_static_model(map_model, f"service_corridor_{index}", [corridor["x"], corridor["y"], 0.021, 0, 0, 0])
        link = ET.SubElement(model, "link", name="link")
        add_box_visual(link, "visual", [0, 0, 0, 0, 0, 0], [corridor["size_x"], corridor["size_y"], 0.01], "0.62 0.62 0.56 1")

    for marker_name, cell, color in (
        ("start_marker", spec["start_cell"], "0.0 0.95 0.15 1"),
        ("goal_marker", spec["goal_cell"], "1.0 0.22 0.0 1"),
    ):
        model = add_static_model(map_model, marker_name, [cell[0] + 0.5, cell[1] + 0.5, 0.035, 0, 0, 0])
        link = ET.SubElement(model, "link", name="link")
        add_cylinder_visual(link, "visual", [0, 0, 0, 0, 0, 0], 0.8, 0.03, color)

    for obstacle in spec["obstacles"]:
        add_equipment(map_model, obstacle)
    add_boundary_fence(map_model, width, height)
    return ET.ElementTree(sdf), origin


def build_obstacle_config(spec, origin):
    return {
        "map_name": spec["map_name"],
        "world_name": spec["world_name"],
        "display_name": spec["display_name"],
        "difficulty": spec["difficulty"],
        "difficulty_label": spec["difficulty_label"],
        "description": spec["description"],
        "note": "Obstacle rectangles use inclusive grid cells; cell centers map to local coordinates using x+0.5 and y+0.5.",
        "gazebo_world_origin_m": origin,
        "width": spec["width"],
        "height": spec["height"],
        "resolution_m": 1.0,
        "start_cell": spec["start_cell"],
        "goal_cell": spec["goal_cell"],
        "altitude_m": 1.5,
        "vertical_safety_margin_m": 0.3,
        "horizontal_inflation_cells": 1,
        "raw_footprint_note": "Obstacle x/y bounds are physical grid footprints before planning inflation.",
        "obstacles": spec["obstacles"],
    }


def generate(output_root):
    output_root = Path(output_root).resolve()
    config_dir = output_root / "config" / "maps"
    world_dir = output_root / "simulation" / "worlds"
    config_dir.mkdir(parents=True, exist_ok=True)
    world_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for spec in MAP_SPECS:
        tree, origin = build_world(spec)
        ET.indent(tree, space="  ")
        world_path = world_dir / f"{spec['world_name']}.sdf"
        tree.write(world_path, encoding="utf-8", xml_declaration=True)

        config = build_obstacle_config(spec, origin)
        config_path = config_dir / f"{spec['world_name']}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        generated.append((world_path, config_path))

    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    for world_path, config_path in generate(args.output_root):
        print(f"Generated {world_path}")
        print(f"Generated {config_path}")


if __name__ == "__main__":
    main()
