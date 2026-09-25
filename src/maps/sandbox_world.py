"""Generate a deterministic Gazebo world from one sandbox map."""

from __future__ import annotations

import math
from xml.etree import ElementTree as ET

from src.maps.sandbox_assets import asset_for
from src.maps.sandbox_contracts import SandboxMap


def _text(parent, tag, value, **attributes):
    node = ET.SubElement(parent, tag, attributes)
    node.text = str(value)
    return node


def _geometry(parent, shape, width, depth, height):
    geometry = ET.SubElement(parent, "geometry")
    if shape == "cylinder":
        cylinder = ET.SubElement(geometry, "cylinder")
        _text(cylinder, "radius", f"{max(width, depth) / 2:g}")
        _text(cylinder, "length", f"{height:g}")
    else:
        box = ET.SubElement(geometry, "box")
        _text(box, "size", f"{width:g} {depth:g} {height:g}")


def _material(parent, color):
    material = ET.SubElement(parent, "material")
    _text(material, "ambient", color)
    _text(material, "diffuse", color)
    _text(material, "specular", "0.08 0.08 0.08 1")


def _add_object(parent, item):
    asset = asset_for(item.asset_id)
    model = ET.SubElement(parent, "model", name=item.object_id)
    _text(model, "static", "true")
    _text(
        model, "pose",
        f"{item.east_m:g} {item.north_m:g} 0 0 0 {math.radians(item.yaw_deg):g}",
    )
    if asset.visual_label is not None:
        plugin = ET.SubElement(
            model, "plugin", filename="gz-sim-label-system",
            name="gz::sim::systems::Label",
        )
        _text(plugin, "label", asset.visual_label)
    link = ET.SubElement(model, "link", name="link")
    pose = f"0 0 {item.height_m / 2:g} 0 0 0"
    collision = ET.SubElement(link, "collision", name="collision")
    _text(collision, "pose", pose)
    _geometry(collision, asset.shape, item.width_m, item.depth_m, item.height_m)
    visual = ET.SubElement(link, "visual", name="visual")
    _text(visual, "pose", pose)
    _geometry(visual, asset.shape, item.width_m, item.depth_m, item.height_m)
    _material(visual, asset.color)


def build_world_tree(map_value: SandboxMap):
    sdf = ET.Element("sdf", version="1.9")
    world = ET.SubElement(sdf, "world", name=f"sandbox_{map_value.map_id}")
    _text(world, "gravity", "0 0 -9.81")
    _text(world, "magnetic_field", "6e-06 2.3e-05 -4.2e-05")
    ET.SubElement(world, "atmosphere", type="adiabatic")
    scene = ET.SubElement(world, "scene")
    # Hide Gazebo's display-only reference grid for presentation flights.
    # This does not alter ground geometry, collisions or planner cells.
    _text(scene, "grid", "false")
    level = map_value.light_level
    _text(scene, "ambient", f"{0.72 * level:g} {0.72 * level:g} {0.72 * level:g} 1")
    _text(scene, "background", "0.72 0.75 0.78 1")
    if map_value.weather == "mist":
        fog = ET.SubElement(scene, "fog")
        _text(fog, "type", "linear")
        _text(fog, "color", "0.78 0.80 0.82 1")
        _text(fog, "start", "8")
        _text(fog, "end", "55")
    light = ET.SubElement(world, "light", name="sun", type="directional")
    _text(light, "cast_shadows", "true")
    _text(light, "pose", "0 0 20 0 0 0")
    diffuse = 0.85 * (0.65 if map_value.weather == "overcast" else level)
    _text(light, "diffuse", f"{diffuse:g} {diffuse:g} {diffuse:g} 1")
    _text(light, "direction", "-0.5 0.1 -0.9")
    coordinates = ET.SubElement(world, "spherical_coordinates")
    _text(coordinates, "surface_model", "EARTH_WGS84")
    _text(coordinates, "world_frame_orientation", "ENU")
    _text(coordinates, "latitude_deg", "47.397971057728974")
    _text(coordinates, "longitude_deg", "8.546163739800146")
    _text(coordinates, "elevation", "0")
    root = ET.SubElement(world, "model", name="sandbox_map")
    _text(root, "static", "true")
    _text(root, "pose", f"{-map_value.start_east_m:g} {-map_value.start_north_m:g} 0 0 0 0")
    ground = ET.SubElement(root, "model", name="ground")
    _text(ground, "static", "true")
    _text(ground, "pose", f"{map_value.width_m / 2:g} {map_value.height_m / 2:g} 0 0 0 0")
    link = ET.SubElement(ground, "link", name="link")
    collision = ET.SubElement(link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    plane = ET.SubElement(geometry, "plane")
    _text(plane, "normal", "0 0 1")
    _text(plane, "size", f"{map_value.width_m:g} {map_value.height_m:g}")
    visual = ET.SubElement(link, "visual", name="visual")
    _text(visual, "pose", "0 0 -0.01 0 0 0")
    _geometry(visual, "box", map_value.width_m, map_value.height_m, 0.02)
    _material(visual, "0.42 0.43 0.40 1")
    for item in map_value.objects:
        _add_object(root, item)
    ET.indent(sdf, space="  ")
    return sdf


def world_bytes(map_value: SandboxMap):
    return ET.tostring(build_world_tree(map_value), encoding="utf-8", xml_declaration=True)
