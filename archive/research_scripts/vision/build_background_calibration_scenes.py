#!/usr/bin/env python3
"""Build isolated development-only background worlds without canonical edits."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import object_sha256
from src.planner.astar_grid import astar, simplify_grid_path
from src.planner.obstacle_config import build_obstacle_map
from src.vision.collection.route import ObservationWaypoint, VisualRoute

OUTPUT = ROOT / "data/research/background_calibration_v1"
SPECS = (
    ("bg_fence_yard_v1", 790101, "fence", [(4, 8, 17, 18), (12, 16, 19, 20), (20, 21, 5, 10)]),
    ("bg_masonry_lane_v1", 790102, "masonry", [(3, 7, 18, 19), (11, 15, 16, 17), (20, 22, 6, 10)]),
    ("bg_pipe_rack_v1", 790103, "pipe_rack", [(4, 8, 16, 19), (12, 16, 18, 21), (20, 22, 5, 10)]),
)


def text(parent, tag, value):
    ET.SubElement(parent, tag).text = str(value)


def visual(link, name, position, size, color, cylinder=False):
    item = ET.SubElement(link, "visual", name=name)
    text(item, "pose", " ".join(map(str, position)))
    geometry = ET.SubElement(item, "geometry")
    if cylinder:
        shape = ET.SubElement(geometry, "cylinder")
        text(shape, "radius", size[0])
        text(shape, "length", size[1])
    else:
        text(ET.SubElement(geometry, "box"), "size", " ".join(map(str, size)))
    material = ET.SubElement(item, "material")
    text(material, "ambient", " ".join(map(str, (*color, 1))))
    text(material, "diffuse", " ".join(map(str, (*color, 1))))
    text(material, "specular", "0.08 0.08 0.08 1")


def collision(link, size, z):
    item = ET.SubElement(link, "collision", name="conservative_collision")
    text(item, "pose", f"0 0 {z} 0 0 0")
    text(ET.SubElement(ET.SubElement(item, "geometry"), "box"), "size", " ".join(map(str, size)))


def model(parent, name, x, y, z=0):
    item = ET.SubElement(parent, "model", name=name)
    text(item, "static", "true")
    text(item, "pose", f"{x} {y} {z} 0 0 0")
    return ET.SubElement(item, "link", name="link")


def build_scene(scene_id, seed, kind, pads):
    sdf = ET.Element("sdf", version="1.9")
    world = ET.SubElement(sdf, "world", name=scene_id)
    text(world, "gravity", "0 0 -9.81")
    text(world, "magnetic_field", "6e-06 2.3e-05 -4.2e-05")
    ET.SubElement(world, "atmosphere", type="adiabatic")
    scene = ET.SubElement(world, "scene")
    text(scene, "ambient", "0.72 0.72 0.72 1")
    text(scene, "background", "0.72 0.75 0.78 1")
    light = ET.SubElement(world, "light", name="sun", type="directional")
    for key, value in {"pose": "0 0 20 0 0 0", "cast_shadows": "true", "diffuse": "0.85 0.85 0.82 1", "specular": "0.2 0.2 0.2 1", "direction": "-0.5 0.1 -0.9"}.items():
        text(light, key, value)
    spherical = ET.SubElement(world, "spherical_coordinates")
    for key, value in {"surface_model": "EARTH_WGS84", "world_frame_orientation": "ENU", "latitude_deg": "47.397971057728974", "longitude_deg": "8.546163739800146", "elevation": "0"}.items():
        text(spherical, key, value)
    yard = ET.SubElement(world, "model", name="background_calibration_yard")
    text(yard, "static", "true")
    text(yard, "pose", "-12 -12 0 0 0 0")
    floor = model(yard, "paved_ground", 12, 12)
    collision(floor, (28, 28, 0.1), -0.05)
    visual(floor, "base", (0, 0, -0.025, 0, 0, 0), (24, 24, 0.05), (0.39, 0.40, 0.38))
    # Procedural material tiles are scene geometry, not edited training images.
    for x in range(12):
        for y in range(12):
            shade = 0.35 + ((x * 7 + y * 11 + seed) % 9) * 0.035
            visual(floor, f"paver_{x}_{y}", (2*x-11, 2*y-11, 0.008, 0, 0, 0), (1.96, 1.96, 0.016), (shade, shade * 0.97, shade * 0.91))
    obstacles = []
    for i, (x0, x1, y0, y1) in enumerate(pads):
        width, depth = x1-x0+1, y1-y0+1
        cx, cy = (x0+x1+1)/2, (y0+y1+1)/2
        link = model(yard, f"non_target_{kind}_{i}", cx, cy)
        collision(link, (width, depth, 3.0), 1.5)
        obstacles.append({"name": f"non_target_{kind}_{i}", "type": "rect", "x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1, "z_min_m": 0, "z_max_m": 3.0, "visual_category": "background_structure"})
        if kind == "fence":
            for j in range(width*4):
                x = -width/2+0.125+j*0.25
                visual(link, f"picket_{j}", (x, 0, 1.35, 0, 0, 0), (0.055, 0.08, 2.7), (0.23, 0.29, 0.30))
            for j, z in enumerate((0.35, 1.3, 2.45)):
                visual(link, f"rail_{j}", (0, 0, z, 0, 0, 0), (width-0.1, 0.12, 0.12), (0.38, 0.40, 0.39))
        elif kind == "masonry":
            visual(link, "wall", (0, 0, 1.3, 0, 0, 0), (width-0.1, depth-0.1, 2.6), (0.42, 0.36, 0.29))
            for row in range(8):
                for col in range(width*2):
                    x=-width/2+0.25+col*0.5
                    shade=0.36+((row*3+col+seed)%5)*0.06
                    visual(link, f"brick_{row}_{col}", (x, -depth/2+0.035, 0.17+row*0.32, 0, 0, 0), (0.47, 0.05, 0.29), (shade, shade*0.8, shade*0.62))
        else:
            for j, x in enumerate((-width/2+0.25, width/2-0.25)):
                for k, y in enumerate((-depth/2+0.25, depth/2-0.25)):
                    visual(link, f"upright_{j}_{k}", (x,y,1.4,0,0,0), (0.12,0.12,2.8), (0.34,0.37,0.39))
            for j in range(5):
                z=0.55+j*0.45
                visual(link, f"pipe_{j}", (0, -depth/2+0.4+(j%2)*0.5, z, 0, math.pi/2, 0), (0.10+0.02*(j%3), width-0.3), (0.32+j*0.04,0.35,0.39), cylinder=True)
    config = {"map_name": scene_id, "world_name": scene_id, "width":24,"height":24,"resolution_m":1.0,"gazebo_world_origin_m":[-12,-12,0],"start_cell":[0,0],"goal_cell":[17,12],"altitude_m":1.5,"vertical_safety_margin_m":0.3,"horizontal_inflation_cells":1,"obstacles":obstacles}
    nav = build_obstacle_map(config)
    points = [(2,10),(4,11),(6,12),(8,11),(10,10),(12,11),(14,12),(16,11)]
    current=(0,0); waypoints=[]
    for i, cell in enumerate(points):
        path=astar(current,cell,nav["inflated_blocking_cells"],24,24)
        if not path:
            raise ValueError("background route is unreachable")
        target=pads[0 if i<4 else 1]
        tx,ty=(target[0]+target[1]+1)/2,(target[2]+target[3]+1)/2
        yaw=math.degrees(math.atan2(tx-cell[0]-0.5,ty-cell[1]-0.5))%360
        waypoints.append(ObservationWaypoint("complete_distant" if i==0 else f"structure_view_{i}","cruise_distant",cell[0]+0.5,cell[1]+0.5,1.5,yaw,25.0 if i==7 else 3.0,"no_target",tuple(simplify_grid_path(path)[1:])))
        current=cell
    back=astar(current,(0,0),nav["inflated_blocking_cells"],24,24)
    if not back:
        raise ValueError("background return is unreachable")
    route=VisualRoute(scene_id+"_route",scene_id,None,None,(0,0),tuple(waypoints),tuple(simplify_grid_path(back)[1:]))
    ET.indent(sdf, space="  ")
    return ET.tostring(sdf, encoding="utf-8", xml_declaration=True), config, route.to_record()


def main():
    # Construct everything before writing; never overwrite existing evidence.
    generated=[(spec,build_scene(*spec)) for spec in SPECS]
    OUTPUT.mkdir(parents=True,exist_ok=False)
    scenes=[]
    for (scene_id,seed,kind,pads),(world,config,route) in generated:
        folder=OUTPUT/scene_id; folder.mkdir()
        payloads={"world.sdf":world,"obstacles.json":(json.dumps(config,indent=2,sort_keys=True)+"\n").encode(),"route.json":(json.dumps(route,indent=2,sort_keys=True)+"\n").encode()}
        files={}
        for name,payload in payloads.items():
            path=folder/name
            with path.open("xb") as stream:
                stream.write(payload)
            files[name]={"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(payload).hexdigest()}
        scenes.append({"scene_id":scene_id,"seed":seed,"split":"development","kind":kind,"spawn_pose":[-12,-12,0,0,0,0],"files":files,"route_identity":route["route_identity_sha256"],"target_equipment_inventory":[],"inventory_is_not_a_truth_receipt":True})
    manifest={"schema_version":1,"package_id":"background-calibration-v1","status":"generated_not_live_validated","development_only":True,"canonical_catalog_modified":False,"scenes":scenes,"pilot_frames_per_scene":60,"emit_stride":30,"startup_wait_s":140,"review_requirements":{"prearm_pairs":100,"minimum_pairing_rate":0.95,"maximum_p95_skew_ms":33.334,"empty_synchronized_truth_required":True,"visual_equipment_exclusion_review_required":True,"joint_exact_and_near_deduplication_required":True,"minimum_incremental_background_per_scene":20,"training_background_quota":600},"automatic_training":False,"automatic_model_promotion":False}
    manifest["identity"]=object_sha256(manifest)
    with (OUTPUT/"manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(OUTPUT),"scenes":len(scenes),"identity":manifest["identity"],"status":manifest["status"]},indent=2))


if __name__ == "__main__":
    main()
