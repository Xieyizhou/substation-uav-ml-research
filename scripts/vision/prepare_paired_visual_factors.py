"""Freeze the 12-pose, four-variant paired visual-factor diagnostic."""
import copy
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.expansion import CLASSES, target_candidates
from src.vision.canonical.gates import annotation_mode_from_world, validate_view_pose
from src.vision.canonical.plan import read_record, write_record

BASE = ROOT / "data/research/ml_training_recovery_v1/paired-visual-factors-v1"
SOURCE = ROOT / "data/research/canonical_views_v1/complex-occlusion-filtered-plan-v1/plan.json"
ROUND = 201
VARIANTS = ("original", "material", "background", "lighting")
NEUTRAL = "0.350 0.350 0.350 1"


def _material_text(material, text):
    changed = 0
    for tag in ("ambient", "diffuse"):
        node = material.find(tag)
        if node is not None:
            node.text = text; changed += 1
    return changed


def mutate(tree, variant, objects):
    world = tree.getroot().find("world"); changed = []
    if variant == "material":
        targets = {row["name"] for row in objects if row["category"] in CLASSES}
        for model in world.iter("model"):
            if model.get("name") not in targets: continue
            for visual in model.iter("visual"):
                if visual.get("name") not in {"body", "front_panel", "reactor"}: continue
                for material in visual.findall("material"):
                    if _material_text(material, NEUTRAL): changed.append(f"material:{model.get('name')}:{visual.get('name')}")
    elif variant == "background":
        scene = world.find("scene"); background = scene.find("background") if scene is not None else None
        if background is None: raise ValueError("Missing scene background")
        background.text = "0.16 0.22 0.31 1"; changed.append("scene:background")
        for model in world.iter("model"):
            name=model.get("name")
            if name not in {"substation_floor", "substation_floor_grid"}: continue
            color="0.270 0.230 0.190 1" if name == "substation_floor" else "0.100 0.110 0.130 1"
            for material in model.iter("material"):
                if _material_text(material,color): changed.append(f"background:{name}")
    elif variant == "lighting":
        scene=world.find("scene"); ambient=scene.find("ambient") if scene is not None else None
        sun=world.find("light[@name='sun']"); diffuse=sun.find("diffuse") if sun is not None else None
        if ambient is None or diffuse is None: raise ValueError("Missing lighting fields")
        ambient.text="0.38 0.42 0.48 1"; diffuse.text="0.58 0.66 0.78 1"
        changed.extend(("scene:ambient","sun:diffuse"))
    elif variant != "original":
        raise ValueError(f"Unknown variant: {variant}")
    return changed


def _masked_payload(path, variant):
    tree=ET.parse(path); world=tree.getroot().find("world")
    if variant == "material":
        for model in world.iter("model"):
            for visual in model.iter("visual"):
                if visual.get("name") in {"body","front_panel","reactor"}:
                    for material in visual.findall("material"):
                        for tag in ("ambient","diffuse"):
                            node=material.find(tag)
                            if node is not None: node.text="__ALLOWED__"
    elif variant == "background":
        world.find("scene/background").text="__ALLOWED__"
        for model in world.iter("model"):
            if model.get("name") in {"substation_floor","substation_floor_grid"}:
                for material in model.iter("material"):
                    for tag in ("ambient","diffuse"):
                        node=material.find(tag)
                        if node is not None: node.text="__ALLOWED__"
    elif variant == "lighting":
        world.find("scene/ambient").text="__ALLOWED__"; world.find("light[@name='sun']/diffuse").text="__ALLOWED__"
    ET.indent(tree,space="  ")
    return ET.tostring(tree.getroot(),encoding="unicode")


def assert_allowed_world_diff(original, candidate, variant):
    if variant == "original": return
    # Mask the same allowed fields in both trees; every other byte of XML structure/text must match.
    if _masked_payload(original,variant) != _masked_payload(candidate,variant):
        raise ValueError(f"Non-allowed world difference for {variant}")


def choose_poses(source, config):
    rows=target_candidates("complex",source["objects"],config["gazebo_world_origin_m"][:2],(config["width"],config["height"]),ROUND)
    legal=[]
    for row in rows:
        try: validate_view_pose(row,config)
        except ValueError: continue
        legal.append(row)
    chosen=[]
    for category in CLASSES:
        options=sorted((r for r in legal if r["category"]==category),key=lambda r:(r["distance"],r["object_id"],r["bearing"],r["view_id"]))
        if len(options)<3: raise ValueError(f"Not enough legal poses for {category}")
        indexes=(0,len(options)//2,len(options)-1)
        for distance_bin,index in zip(("near","mid","far"),indexes):
            row=copy.deepcopy(options[index]); row["distance_bin"]=distance_bin
            row["source_view_id"]=row["view_id"]
            row["pair_id"]=object_sha256({"round":ROUND,"source_view_id":row["view_id"]})
            chosen.append(row)
    return sorted(chosen,key=lambda r:(r["category"],r["object_id"],r["bearing"],r["view_id"]))


def prepare(base=BASE):
    base=Path(base)
    if (base/"protocol.json").exists(): return json.loads((base/"protocol.json").read_text())
    source=read_record(SOURCE); config=json.loads((SOURCE.parent/"obstacles.json").read_text())
    poses=choose_poses(source,config); plans={}; worlds={}
    for variant in VARIANTS:
        folder=base/variant/"plan"; folder.mkdir(parents=True,exist_ok=False)
        for name,digest in source["files"].items():
            src=SOURCE.parent/name
            if file_sha256(src)!=digest: raise ValueError(f"Source artifact changed: {src}")
            (folder/name).write_bytes(src.read_bytes())
        world_path=folder/"world.sdf"; tree=ET.parse(world_path)
        sensor=[s for s in tree.iter("sensor") if s.get("type")=="boundingbox_camera"]
        if len(sensor)!=1 or sensor[0].find("camera/box_type") is None: raise ValueError("Invalid camera declaration")
        sensor[0].find("camera/box_type").text="full_2d"
        changed=mutate(tree,variant,source["objects"]); ET.indent(tree,space="  ")
        tree.write(world_path,encoding="utf-8",xml_declaration=True)
        if annotation_mode_from_world(world_path)!="full_2d": raise ValueError("box_type did not persist")
        plan={k:copy.deepcopy(v) for k,v in source.items() if k!="identity"}
        variant_views=[]
        for pose in poses:
            row=copy.deepcopy(pose); row["variant"]=variant
            # view_id remains the frozen pose identity shared by all four variants.
            variant_views.append(row)
        plan.update(annotation_mode="full_2d",label_mode="visual-instance",hierarchy_mode="top-level-equipment",
                    calibration_views=variant_views,pilot_views=[],diagnostic_only=True,
                    diagnostic_require_expected_presence=True,diagnostic_allow_expected_absence=False,
                    experiment_id="paired-visual-factors-v1",variant=variant,variant_changes=changed,
                    expansion_round=ROUND,training_admitted=False,promotable=False,automatic_training=False)
        plan["files"]["world.sdf"]=file_sha256(world_path)
        plans[variant]=write_record(folder/"plan.json",plan); worlds[variant]=world_path
    for variant in VARIANTS[1:]: assert_allowed_world_diff(worlds["original"],worlds[variant],variant)
    protocol={"schema_version":1,"status":"frozen_pending_capture","experiment_id":"paired-visual-factors-v1",
              "development_diagnostic_group":"paired-visual-factors-v1","scene":"complex","expansion_round":ROUND,
              "annotation":{"box_type":"full_2d","label_mode":"visual-instance","hierarchy_mode":"top-level-equipment"},
              "poses":[{k:r[k] for k in ("pair_id","source_view_id","category","object_id","bearing","distance","distance_bin","height","offset","position","orientation","camera_position")} for r in poses],
              "variants":[{"variant":v,"plan_path":str(base/v/"plan/plan.json"),"plan_identity":plans[v]["identity"],"world_sha256":plans[v]["files"]["world.sdf"]} for v in VARIANTS],
              "capture":{"technical_retries":3,"pose_tolerance_m":.05,"attitude_tolerance_deg":1,"max_skew_ms":33.334,"stable_frames":3},
              "evaluation":{"input_size":640,"device":"cpu","confidence":.37,"class_aware_nms_iou":.7,"max_det":300,"match_iou":.5},
              "review_nature":"AI-assisted per-frame review","training_admitted":False,"promotable":False,
              "limits":["Development-only; no protected labels, training, threshold tuning or promotion.","Background combines sky, ground and grid changes; their individual contributions are not identifiable.","Three poses per class support directional diagnosis in this scene only."]}
    protocol["identity"]=object_sha256(protocol); write_json(base/"protocol.json",protocol)
    return protocol


if __name__=="__main__": print(json.dumps(prepare(),indent=2,ensure_ascii=False))
