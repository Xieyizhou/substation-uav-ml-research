"""Freeze the 240-frame bounded augmentation matrix and its data-role ledger."""
import copy
import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.expansion import CLASSES, _pose, target_candidates
from src.vision.canonical.gates import validate_view_pose
from src.vision.canonical.occlusion import view_blockers
from src.vision.canonical.plan import read_record, write_record

BASE=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-240-v1"
SOURCE=ROOT/"data/research/canonical_views_v1/complex-occlusion-filtered-plan-v1/plan.json"
CONFIG=ROOT/"config/maps/substation_complex.json"
TARGET_ROUND=211; REGULAR_ROUND=221; NEGATIVE_ROUND=231
MATERIALS={"mat_cool_gray":"0.35 0.37 0.40 1","mat_warm_oxide":"0.43 0.31 0.23 1","mat_desaturated_green":"0.28 0.38 0.32 1"}
LIGHTS={"light_normal":None,"light_cool_low":("0.30 0.34 0.40 1","0.48 0.56 0.68 1")}


def known_ids():
    result=set()
    for path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(path): continue
        try: data=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError): continue
        result.update(r.get("view_id") for r in data.get("views",[]) if r.get("view_id"))
    return result


def spread(rows,count):
    rows=sorted(rows,key=lambda r:(r["distance"],r["object_id"],r["bearing"],r["height"],r["view_id"]))
    if len(rows)<count: raise ValueError(f"Need {count} candidates, found {len(rows)}")
    picked=[rows[min(len(rows)-1,int(i*len(rows)/count))] for i in range(count)]
    if len({r["view_id"] for r in picked})!=count: raise ValueError("Spread produced duplicate poses")
    return [copy.deepcopy(r) for r in picked]


def negative_candidates(source,config):
    non_targets=[o for o in source["objects"] if o["category"] not in CLASSES]
    rows=[]; origin=config["gazebo_world_origin_m"][:2]
    for oi,obj in enumerate(non_targets):
        b=obj["bounds"]; target=[(b[0]+b[1])/2,(b[2]+b[3])/2,(b[4]+b[5])/2]
        for step in range(16):
            bearing=(step*45+oi*17+NEGATIVE_ROUND)%360; distance=(4.5,6.0,7.5,9.0)[step%4]; height=(1.8,2.5,3.2)[step%3]
            a=math.radians(bearing); camera=[target[0]+distance*math.cos(a),target[1]+distance*math.sin(a),height]
            position,orientation=_pose(camera,target,0)
            row={"map_id":"complex","object_id":obj["name"],"category":"no_target","hard_negative_type":obj["category"],"bearing":bearing,"distance":distance,"height":height,"offset":0,"camera_position":camera,"position":position,"orientation":orientation,"family":f"complex:negative:{obj['name']}:{bearing//45}"}
            row["view_id"]=object_sha256(row)
            try: validate_view_pose(row,config)
            except ValueError: continue
            if not view_blockers(row,source["objects"]): rows.append(row)
    rows=sorted(rows,key=lambda r:(r["hard_negative_type"],r["object_id"],r["bearing"],r["view_id"]))
    return spread(rows,24)


def set_materials(world,material_id,objects):
    if material_id=="material_original": return
    palette=list(MATERIALS); shift=palette.index(material_id); categories=list(CLASSES)
    assignment={category:palette[(i+shift)%len(palette)] for i,category in enumerate(categories)}
    by_name={o["name"]:o["category"] for o in objects}
    for model in world.iter("model"):
        category=by_name.get(model.get("name")); selected=assignment.get(category)
        if selected is None: continue
        for visual in model.iter("visual"):
            if visual.get("name") not in {"body","front_panel","reactor"}: continue
            for material in visual.findall("material"):
                for tag in ("ambient","diffuse"):
                    node=material.find(tag)
                    if node is not None: node.text=MATERIALS[selected]


def set_light(world,light_id):
    values=LIGHTS[light_id]
    if values is None:return
    ambient=world.find("scene/ambient"); diffuse=world.find("light[@name='sun']/diffuse")
    if ambient is None or diffuse is None:raise ValueError("Missing lighting fields")
    ambient.text,diffuse.text=values


def make_plan(source,views,run_id,material_id,light_id,negative=False):
    folder=BASE/"runs"/run_id/"plan"; folder.mkdir(parents=True,exist_ok=False)
    for name,digest in source["files"].items():
        src=SOURCE.parent/name
        if file_sha256(src)!=digest:raise ValueError(f"Source artifact changed: {src}")
        (folder/name).write_bytes(src.read_bytes())
    tree=ET.parse(folder/"world.sdf"); world=tree.getroot().find("world")
    sensor=[s for s in tree.iter("sensor") if s.get("type")=="boundingbox_camera"]
    if len(sensor)!=1:raise ValueError("Expected one bounding-box camera")
    sensor[0].find("camera/box_type").text="full_2d"; set_materials(world,material_id,source["objects"]); set_light(world,light_id)
    ET.indent(tree,space="  "); tree.write(folder/"world.sdf",encoding="utf-8",xml_declaration=True)
    plan={k:copy.deepcopy(v) for k,v in source.items() if k!="identity"}
    plan.update(annotation_mode="full_2d",label_mode="visual-instance",hierarchy_mode="top-level-equipment",calibration_views=views,pilot_views=[],diagnostic_only=True,diagnostic_require_expected_presence=not negative,diagnostic_allow_expected_absence=negative,diagnostic_require_no_targets=negative,training_admitted=False,promotable=False,automatic_training=False,diagnostic_purpose="visual_augmentation_240_candidate",material_id=material_id,lighting_id=light_id,recording_group=run_id)
    plan["files"]["world.sdf"]=file_sha256(folder/"world.sdf")
    return write_record(folder/"plan.json",plan),folder/"world.sdf"


def prepare():
    if (BASE/"matrix.json").exists():return json.loads((BASE/"matrix.json").read_text())
    source=read_record(SOURCE); config=json.loads(CONFIG.read_text()); known=known_ids()
    candidates=[r for r in target_candidates("complex",source["objects"],config["gazebo_world_origin_m"][:2],(config["width"],config["height"]),TARGET_ROUND) if r["view_id"] not in known]
    augmented=[]
    for category in CLASSES: augmented.extend(spread([r for r in candidates if r["category"]==category],6))
    regular_candidates=[r for r in target_candidates("complex",source["objects"],config["gazebo_world_origin_m"][:2],(config["width"],config["height"]),REGULAR_ROUND) if r["view_id"] not in known and r["view_id"] not in {x["view_id"] for x in augmented}]
    regular=[]
    for category in CLASSES:regular.extend(spread([r for r in regular_candidates if r["category"]==category],12))
    negatives=negative_candidates(source,config); runs=[]; ledger=[]
    specifications=[]
    for material in MATERIALS:
        for light in LIGHTS: specifications.append((f"appearance-{material}-{light}",augmented,material,light,False,"appearance_lighting_positive"))
    for light in LIGHTS:specifications.append((f"hard-negative-{light}",negatives,"material_original",light,True,"hard_negative"))
    specifications.append(("regular-positive",regular,"material_original","light_normal",False,"regular_positive"))
    for run_id,base_views,material,light,negative,subset in specifications:
        views=[]
        for base_view in base_views:
            row=copy.deepcopy(base_view); row["pose_id"]=base_view["view_id"]; row["derivation_group"]=f"pose:{base_view['view_id']}"; row["material_id"]=material; row["lighting_id"]=light; row["subset"]=subset; views.append(row)
        plan,world=make_plan(source,views,run_id,material,light,negative)
        runs.append({"run_id":run_id,"subset":subset,"frame_count":len(views),"plan_path":str(BASE/"runs"/run_id/"plan/plan.json"),"plan_identity":plan["identity"],"world_sha256":file_sha256(world),"material_id":material,"lighting_id":light,"require_no_targets":negative})
        for row in views:
            cid=object_sha256({"run":run_id,"pose":row["pose_id"]})
            ledger.append({"candidate_id":cid,"data_role":"new_training_candidate","map_layout_id":"complex-canonical-v1","asset_family_ids":["canonical-complex-equipment-v1"],"equipment_instance_id":None if negative else row["object_id"],"recording_group":run_id,"derivation_group":row["derivation_group"],"pose_id":row["pose_id"],"material_id":material,"lighting_id":light,"annotation_mode":"full_2d","image_sha256":None,"label_sha256":None,"world_sha256":file_sha256(world),"sensor_sha256":plan["files"]["sensor_source.sdf"],"instance_present":None,"visibility_status":"unknown" if not negative else "not_applicable","truncation_status":"unknown" if not negative else "not_applicable","capture_status":"planned","review_decision":"pending","review_reason":None,"review_nature":"pending","reviewed_at":None,"exact_duplicate_of":None,"pixel_duplicate_of":None,"lineage_relations":[row["derivation_group"]],"dedup_status":"designed_variant" if subset=="appearance_lighting_positive" else "pending","admission_status":"pending","failure_reason":None})
    if len(ledger)!=240 or sum(r["frame_count"] for r in runs)!=240:raise ValueError("Matrix is not exactly 240 frames")
    matrix={"schema_version":1,"status":"frozen_pending_pilot","dataset_version":"visual-augmentation-240-v1","counts":{"appearance_lighting_positive":144,"hard_negative":48,"regular_positive":48,"total":240},"independent_pose_groups":{"appearance_lighting_positive":24,"hard_negative":24,"regular_positive":48},"runs":runs,"selection":{"appearance_round":TARGET_ROUND,"regular_round":REGULAR_ROUND,"negative_round":NEGATIVE_ROUND,"known_view_ids_excluded":len(known)},"training_admitted":False,"promotable":False}; matrix["identity"]=object_sha256(matrix)
    intake={"schema_version":1,"dataset_version":"visual-augmentation-240-v1","status":"frozen_pending_capture","entries":ledger,"training_admitted":False,"promotable":False}; intake["identity"]=object_sha256(intake)
    roles={"schema_version":1,"roles":{"trusted_training_base":{"rule":"Common basis for all arms; only uniformly reviewed admitted members."},"new_training_candidate":{"rule":"Pending unified capture, review, dedup and lineage admission."},"fixed_development_regression":{"path":"data/research/ml_training_recovery_v1/paired-visual-factors-v1","rule":"48 reviewed frames; never train; not blind."},"frozen_unseen_scene_test":{"path":"data/research/ml_training_recovery_v1/visual-augmentation-240-v1/unseen-scene-test-plan.json","rule":"Sealed before training; evaluate only after development candidate selection."}},"partition_keys":["map_layout_id","asset_family_ids","equipment_instance_id","recording_group","derivation_group"],"training_admitted":False,"promotable":False};roles["identity"]=object_sha256(roles)
    BASE.mkdir(parents=True,exist_ok=True);write_json(BASE/"matrix.json",matrix);write_json(BASE/"intake-ledger.json",intake);write_json(BASE/"data-roles.json",roles)
    return matrix


if __name__=="__main__":print(json.dumps(prepare(),indent=2,ensure_ascii=False))
