#!/usr/bin/env python3
"""Add the repository-owned research vehicle to a runtime Gazebo world."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ElementTree


def pose_text(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise ValueError("spawn pose must contain x,y,z,roll,pitch,yaw")
    numbers = [float(part) for part in parts]
    return " ".join(f"{number:g}" for number in numbers)


def prepare_world_with_vehicle(source, output, *, model_name, entity_name, pose):
    tree = ElementTree.parse(source)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("SDF file does not contain a world")
    if any(model.get("name") == entity_name for model in world.findall("model")):
        raise ValueError(f"world already contains model {entity_name}")
    include = ElementTree.Element("include")
    ElementTree.SubElement(include, "uri").text = f"model://{model_name}"
    ElementTree.SubElement(include, "name").text = entity_name
    ElementTree.SubElement(include, "pose").text = pose_text(pose)
    world.append(include)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="x500_research")
    parser.add_argument("--name", default="x500_research_0")
    parser.add_argument("--pose", required=True)
    args = parser.parse_args(argv)
    try:
        prepare_world_with_vehicle(
            args.source,
            args.output,
            model_name=args.model,
            entity_name=args.name,
            pose=args.pose,
        )
    except (ElementTree.ParseError, OSError, ValueError) as error:
        print(f"Could not prepare research vehicle: {error}")
        return 1
    print(f"Prepared research vehicle: {args.name} ({args.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
