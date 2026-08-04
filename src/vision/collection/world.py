"""Gazebo SDF v3 materialization for deterministic visual layouts."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from src.vision.collection.layout import LABEL_BY_CLASS, LayoutManifest


COLORS = {
    "transformer": (0.12, 0.28, 0.34),
    "switchgear": (0.10, 0.42, 0.52),
    "capacitor_bank": (0.30, 0.48, 0.48),
    "reactor": (0.38, 0.38, 0.42),
    "cabinet": (0.08, 0.32, 0.40),
    "pole": (0.12, 0.12, 0.12),
    "control_building": (0.38, 0.38, 0.36),
    "unknown_obstacle": (0.65, 0.18, 0.10),
}
BACKGROUND_BY_WEATHER = {
    "clear": "0.72 0.76 0.82 1",
    "overcast": "0.50 0.53 0.56 1",
    "mist": "0.68 0.70 0.70 1",
}


def _text(parent, tag, value, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    element.text = str(value)
    return element


def _pose(*values):
    return " ".join(f"{float(value):.6g}" for value in values)


def _box(parent, size):
    geometry = ET.SubElement(parent, "geometry")
    box = ET.SubElement(geometry, "box")
    _text(box, "size", _pose(*size))


def _cylinder(parent, radius, length):
    geometry = ET.SubElement(parent, "geometry")
    cylinder = ET.SubElement(geometry, "cylinder")
    _text(cylinder, "radius", f"{radius:.6g}")
    _text(cylinder, "length", f"{length:.6g}")


def _material(visual, category, age):
    base = COLORS[category]
    factor = 1.0 - 0.45 * float(age)
    color = " ".join(f"{channel * factor:.4f}" for channel in base) + " 1"
    material = ET.SubElement(visual, "material")
    _text(material, "ambient", color)
    _text(material, "diffuse", color)
    _text(material, "specular", "0.06 0.06 0.06 1")


def _add_geometry(link, item):
    height = item.height_m
    center_z = height / 2
    for kind in ("collision", "visual"):
        element = ET.SubElement(link, kind, name=kind)
        _text(element, "pose", _pose(0, 0, center_z, 0, 0, 0))
        if item.visual_category in {"reactor", "pole"}:
            radius = 0.2 if item.visual_category == "pole" else min(item.size_east_m, item.size_north_m) * 0.42
            _cylinder(element, radius, height)
        else:
            _box(element, (item.size_east_m, item.size_north_m, height))
        if kind == "visual":
            _material(element, item.visual_category, item.material_age)
            if item.simulator_label is not None:
                plugin = ET.SubElement(
                    element,
                    "plugin",
                    filename="gz-sim-label-system",
                    name="gz::sim::systems::Label",
                )
                _text(plugin, "label", item.simulator_label)


def _add_object(parent, item, *, origin_east_m=0.0, origin_north_m=0.0):
    model = ET.SubElement(parent, "model", name=item.object_id)
    _text(model, "static", "true")
    _text(
        model,
        "pose",
        _pose(
            item.east_m + origin_east_m,
            item.north_m + origin_north_m,
            0,
            0,
            0,
            math.radians(item.yaw_deg),
        ),
    )
    link = ET.SubElement(model, "link", name="link")
    _add_geometry(link, item)


def build_layout_world(manifest):
    if not isinstance(manifest, LayoutManifest):
        raise TypeError("manifest must be a LayoutManifest")
    sdf = ET.Element("sdf", version="1.9")
    world = ET.SubElement(sdf, "world", name=f"visual_{manifest.layout_id}")
    _text(world, "gravity", "0 0 -9.81")
    _text(world, "magnetic_field", "6e-06 2.3e-05 -4.2e-05")
    ET.SubElement(world, "atmosphere", type="adiabatic")
    scene = ET.SubElement(world, "scene")
    background = BACKGROUND_BY_WEATHER[manifest.weather]
    _text(scene, "ambient", background)
    _text(scene, "background", background)
    light = ET.SubElement(world, "light", name="sun", type="directional")
    _text(light, "cast_shadows", "true")
    _text(light, "pose", "0 0 25 0 0 0")
    diffuse = min(1.0, 0.8 * manifest.light_intensity)
    _text(light, "diffuse", _pose(diffuse, diffuse, diffuse, 1))
    _text(light, "direction", "-0.5 0.1 -0.9")
    spherical = ET.SubElement(world, "spherical_coordinates")
    _text(spherical, "surface_model", "EARTH_WGS84")
    _text(spherical, "world_frame_orientation", "ENU")
    _text(spherical, "latitude_deg", "47.397971057728974")
    _text(spherical, "longitude_deg", "8.546163739800146")
    _text(spherical, "elevation", "0")
    station = ET.SubElement(world, "model", name="substation_map")
    _text(station, "static", "true")
    _text(station, "pose", _pose(-manifest.width_m / 2, -manifest.height_m / 2, 0, 0, 0, 0))
    ground = ET.SubElement(station, "model", name="ground_plane")
    _text(ground, "static", "true")
    _text(ground, "pose", _pose(manifest.width_m / 2, manifest.height_m / 2, 0, 0, 0, 0))
    link = ET.SubElement(ground, "link", name="link")
    collision = ET.SubElement(link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    plane = ET.SubElement(geometry, "plane")
    _text(plane, "normal", "0 0 1")
    _text(plane, "size", _pose(manifest.width_m + 4, manifest.height_m + 4))
    floor = ET.SubElement(station, "model", name="substation_floor")
    _text(floor, "static", "true")
    _text(floor, "pose", _pose(manifest.width_m / 2, manifest.height_m / 2, 0.01, 0, 0, 0))
    floor_link = ET.SubElement(floor, "link", name="link")
    visual = ET.SubElement(floor_link, "visual", name="visual")
    _box(visual, (manifest.width_m, manifest.height_m, 0.02))
    _material(visual, "control_building", 0.3)
    for item in manifest.objects:
        _add_object(
            world,
            item,
            origin_east_m=-manifest.width_m / 2,
            origin_north_m=-manifest.height_m / 2,
        )
    return ET.ElementTree(sdf)


def write_layout_world(path, manifest):
    tree = build_layout_world(manifest)
    ET.indent(tree, space="  ")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _set_camera_pitch(tree, pitch_down_deg):
    camera_link = tree.find(".//link[@name='research_camera_link']")
    if camera_link is None:
        raise ValueError("research camera link was not found")
    link_pose = camera_link.find("pose")
    pose_values = [] if link_pose is None else link_pose.text.split()
    if len(pose_values) != 6:
        raise ValueError("research camera link pose is invalid")
    pose_values[4] = f"{math.radians(float(pitch_down_deg)):.8g}"
    link_pose.text = " ".join(pose_values)


def _set_camera_intrinsics(camera):
    width = int(camera.findtext("image/width"))
    height = int(camera.findtext("image/height"))
    horizontal_fov = float(camera.findtext("horizontal_fov"))
    focal_length = width / (2.0 * math.tan(horizontal_fov / 2.0))
    lens = camera.find("lens")
    if lens is None:
        lens = ET.SubElement(camera, "lens")
    existing = lens.find("intrinsics")
    if existing is not None:
        lens.remove(existing)
    intrinsics = ET.SubElement(lens, "intrinsics")
    values = (
        ("fx", focal_length), ("fy", focal_length),
        ("cx", width / 2.0), ("cy", height / 2.0), ("s", 0.0),
    )
    for tag, value in values:
        _text(intrinsics, tag, f"{value:.8g}")


def materialize_camera_model(
    source_path,
    output_path,
    noise_stddev,
    *,
    pitch_down_deg=0.0,
):
    tree = ET.parse(source_path)
    _set_camera_pitch(tree, pitch_down_deg)
    cameras = [
        tree.find(f".//sensor[@name='{name}']/camera")
        for name in ("research_rgb", "research_boxes")
    ]
    if any(camera is None for camera in cameras):
        raise ValueError("research RGB and truth cameras were not found")
    for camera in cameras:
        _set_camera_intrinsics(camera)
    camera = cameras[0]
    existing = camera.find("noise")
    if existing is not None:
        camera.remove(existing)
    noise = ET.SubElement(camera, "noise")
    _text(noise, "type", "gaussian")
    _text(noise, "mean", "0")
    _text(noise, "stddev", f"{float(noise_stddev):.8g}")
    ET.indent(tree, space="  ")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def validate_world_matches_layout(tree, manifest):
    root = tree.getroot() if isinstance(tree, ET.ElementTree) else tree
    station = root.find("./world/model[@name='substation_map']")
    if station is None:
        raise ValueError("visual world has no substation_map")
    models = {model.get("name"): model for model in root.findall("./world/model")}
    for item in manifest.objects:
        model = models.get(item.object_id)
        if model is None:
            raise ValueError(f"visual world is missing {item.object_id}")
        pose = [float(value) for value in model.findtext("./pose").split()]
        expected_xy = (
            item.east_m - manifest.width_m / 2,
            item.north_m - manifest.height_m / 2,
        )
        if len(pose) != 6 or any(
            not math.isclose(actual, expected, abs_tol=1e-4)
            for actual, expected in zip(pose[:2], expected_xy)
        ):
            raise ValueError(f"visual world pose mismatch for {item.object_id}")
        label = model.findtext("./link/visual/plugin/label")
        expected = LABEL_BY_CLASS.get(item.visual_category)
        if label != (None if expected is None else str(expected)):
            raise ValueError(f"visual world label mismatch for {item.object_id}")
    return True
