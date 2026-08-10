"""Deterministic equipment-centered layout manifests for visual collection v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256
from src.maps.map_catalog import project_path


LAYOUT_SCHEMA_VERSION = 1
LABEL_BY_CLASS = {name: index + 1 for index, name in enumerate(EQUIPMENT_CLASSES)}
BASE_OBJECTS = (
    ("transformer_a", "transformer", 18.0, 8.0, 1.8, 1.8, 2.1),
    ("transformer_b", "transformer", 34.0, 32.0, 2.4, 2.4, 2.3),
    ("switchgear_a", "switchgear", 26.0, 8.0, 1.6, 1.4, 1.7),
    ("switchgear_b", "switchgear", 26.0, 32.0, 2.2, 1.8, 1.9),
    ("capacitor_bank_a", "capacitor_bank", 34.0, 8.0, 1.7, 1.7, 1.9),
    ("capacitor_bank_b", "capacitor_bank", 18.0, 32.0, 2.3, 2.3, 2.1),
    ("reactor_a", "reactor", 34.0, 20.0, 1.6, 1.6, 2.3),
    ("reactor_b", "reactor", 18.0, 20.0, 2.2, 2.2, 2.5),
    ("cabinet", "cabinet", 22.0, 20.0, 1.5, 2.0, 1.5),
    ("pole", "pole", 30.0, 20.0, 0.5, 0.5, 4.8),
    ("control_building", "control_building", 8.0, 20.0, 2.0, 32.0, 3.0),
)


@dataclass(frozen=True)
class LayoutObject:
    object_id: str
    visual_category: str
    east_m: float
    north_m: float
    size_east_m: float
    size_north_m: float
    height_m: float
    yaw_deg: float
    material_age: float
    simulator_label: int | None


@dataclass(frozen=True)
class LayoutManifest:
    layout_id: str
    split: str
    layout_seed: int
    width_m: float
    height_m: float
    start_cell: tuple[int, int]
    weather: str
    light_intensity: float
    camera_noise_stddev: float
    objects: tuple[LayoutObject, ...]
    generator_version: str = "gazebo-sdf-v3"
    layout_schema_version: int = LAYOUT_SCHEMA_VERSION

    def identity_record(self):
        record = asdict(self)
        record["start_cell"] = list(self.start_cell)
        return record

    @property
    def layout_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "layout_identity_sha256": self.layout_identity_sha256}


def _ranges(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if values.get("schema_version") != 2:
        raise ValueError("unsupported visual layout randomization schema")
    return values


def _sample_object(generator, ranges, specification):
    object_id, category, east, north, size_east, size_north, height = specification
    is_target = category in EQUIPMENT_CLASSES
    jitter = abs(float(ranges["equipment_position_jitter_m"][1])) if is_target else 0.0
    scale = generator.uniform(*map(float, ranges["equipment_scale"])) if is_target else 1.0
    sampled_east_m = east + generator.uniform(-jitter, jitter)
    sampled_north_m = north + generator.uniform(-jitter, jitter)
    sampled_yaw_deg = float(generator.choice(ranges["equipment_yaw_deg"]))
    material_age = generator.uniform(*map(float, ranges["material_age"]))
    return LayoutObject(
        object_id=object_id,
        visual_category=category,
        east_m=sampled_east_m,
        north_m=sampled_north_m,
        size_east_m=size_east * scale,
        size_north_m=size_north * scale,
        height_m=height * scale,
        yaw_deg=sampled_yaw_deg if is_target else 0.0,
        material_age=material_age,
        simulator_label=LABEL_BY_CLASS.get(category),
    )


def _unknown_objects(generator, ranges):
    count = generator.randint(*map(int, ranges["unknown_obstacle_count"]))
    anchors = ((12.0, 6.0), (12.0, 20.0), (12.0, 34.0))
    objects = []
    for index in range(count):
        size = generator.uniform(*map(float, ranges["unknown_obstacle_size_m"]))
        east, north = anchors[index]
        objects.append(
            LayoutObject(
                object_id=f"unknown_obstacle_{index + 1:02d}",
                visual_category="unknown_obstacle",
                east_m=east,
                north_m=north,
                size_east_m=size,
                size_north_m=size,
                height_m=generator.uniform(0.8, 2.4),
                yaw_deg=0.0,
                material_age=generator.uniform(*map(float, ranges["material_age"])),
                simulator_label=None,
            )
        )
    return objects


def build_layout_manifest(layout_id, split, layout_seed, randomization_path):
    ranges = _ranges(project_path(randomization_path))
    generator = random.Random(int(layout_seed))
    objects = [_sample_object(generator, ranges, item) for item in BASE_OBJECTS]
    objects.extend(_unknown_objects(generator, ranges))
    manifest = LayoutManifest(
        layout_id=str(layout_id),
        split=str(split),
        layout_seed=int(layout_seed),
        width_m=40.0,
        height_m=40.0,
        start_cell=(2, 2),
        weather=generator.choice(tuple(ranges["weather"])),
        light_intensity=generator.uniform(*map(float, ranges["light_intensity"])),
        camera_noise_stddev=generator.uniform(*map(float, ranges["camera_noise_stddev"])),
        objects=tuple(objects),
    )
    validate_layout_manifest(manifest)
    return manifest


def _bounds(item, margin=0.0):
    angle = math.radians(item.yaw_deg)
    cosine = abs(math.cos(angle))
    sine = abs(math.sin(angle))
    cosine = 0.0 if cosine < 1e-12 else cosine
    sine = 0.0 if sine < 1e-12 else sine
    half_east = (
        cosine * item.size_east_m + sine * item.size_north_m
    ) / 2
    half_north = (
        sine * item.size_east_m + cosine * item.size_north_m
    ) / 2
    return (
        item.east_m - half_east - margin,
        item.north_m - half_north - margin,
        item.east_m + half_east + margin,
        item.north_m + half_north + margin,
    )


def _overlap(first, second):
    return not (
        first[2] <= second[0] or second[2] <= first[0]
        or first[3] <= second[1] or second[3] <= first[1]
    )


def validate_layout_manifest(manifest):
    if manifest.generator_version != "gazebo-sdf-v3":
        raise ValueError("unsupported visual layout generator")
    counts = {class_name: 0 for class_name in EQUIPMENT_CLASSES}
    for item in manifest.objects:
        bounds = _bounds(item)
        if bounds[0] < 1 or bounds[1] < 1 or bounds[2] > manifest.width_m - 1 or bounds[3] > manifest.height_m - 1:
            raise ValueError(f"layout object outside boundary: {item.object_id}")
        expected_label = LABEL_BY_CLASS.get(item.visual_category)
        if item.simulator_label != expected_label:
            raise ValueError(f"layout label mismatch: {item.object_id}")
        if item.visual_category in counts:
            counts[item.visual_category] += 1
    if any(count < 2 for count in counts.values()):
        raise ValueError("each visual class requires at least two objects")
    for index, first in enumerate(manifest.objects):
        for second in manifest.objects[index + 1:]:
            if _overlap(_bounds(first, 0.3), _bounds(second, 0.3)):
                raise ValueError(f"layout objects overlap: {first.object_id}, {second.object_id}")
    return True


def layout_obstacle_config(manifest):
    obstacles = []
    for item in manifest.objects:
        x_min, y_min, x_max, y_max = _bounds(item)
        obstacles.append(
            {
                "name": item.object_id,
                "type": "rect",
                "x_min": int(x_min),
                "x_max": int(x_max),
                "y_min": int(y_min),
                "y_max": int(y_max),
                "z_min_m": 0.0,
                "z_max_m": item.height_m,
                "visual_category": item.visual_category,
            }
        )
    return {
        "map_name": f"visual_{manifest.layout_id}",
        "world_name": f"visual_{manifest.layout_id}",
        "width": int(manifest.width_m),
        "height": int(manifest.height_m),
        "resolution_m": 1.0,
        "start_cell": list(manifest.start_cell),
        "goal_cell": list(manifest.start_cell),
        "gazebo_world_origin_m": [-manifest.width_m / 2, -manifest.height_m / 2, 0.0],
        "altitude_m": 1.5,
        "vertical_safety_margin_m": 0.3,
        "horizontal_inflation_cells": 1,
        "obstacles": obstacles,
    }
