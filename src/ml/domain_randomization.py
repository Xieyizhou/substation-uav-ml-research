"""Deterministic simulator domain-randomization manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path


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

    return {
        "schema_version": 1,
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
        "weather": generator.choice(list(config["weather"])),
    }
