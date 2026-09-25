"""Build an isolated, lineage-bound 24-pose x 2-light hard-negative matrix."""
import copy
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,write_record
from scripts.vision.prepare_visual_augmentation_batch import (
    BASE as V1_BASE, CONFIG, LIGHTS, SOURCE, negative_candidates, set_light,
)

BASE=ROOT/"data/research/ml_training_recovery_v1/hard-negative-isolated-v2"
TARGETS={"transformer","switchgear","capacitor_bank","reactor"}


def isolate_world(path,source):
    tree=ET.parse(path);world=tree.getroot().find("world")
    target_names={row["name"] for row in source["objects"] if row["category"] in TARGETS}
    removed=[]
    for model in list(world.findall("model")):
        if model.get("name") in target_names:
            removed.append(model.get("name"));world.remove(model)
    if set(removed)!=target_names:raise ValueError("Did not isolate every configured target model")
    ET.indent(tree,space="  ");tree.write(path,encoding="utf-8",xml_declaration=True)
    return sorted(removed)


def make_plan(source,views,run_id,light_id):
    folder=BASE/"runs"/run_id/"plan";folder.mkdir(parents=True,exist_ok=False)
    for name,digest in source["files"].items():
        src=SOURCE.parent/name
        if file_sha256(src)!=digest:raise ValueError(f"Source artifact changed: {src}")
        (folder/name).write_bytes(src.read_bytes())
    world_path=folder/"world.sdf";tree=ET.parse(world_path)
    sensor=[s for s in tree.iter("sensor") if s.get("type")=="boundingbox_camera"]
    if len(sensor)!=1 or sensor[0].find("camera/box_type") is None:raise ValueError("Invalid camera declaration")
    sensor[0].find("camera/box_type").text="full_2d";world=tree.getroot().find("world")
    targets={row["name"] for row in source["objects"] if row["category"] in TARGETS};removed=[]
    for model in list(world.findall("model")):
        if model.get("name") in targets:removed.append(model.get("name"));world.remove(model)
    if set(removed)!=targets:raise ValueError("Target isolation mismatch")
    set_light(world,light_id);ET.indent(tree,space="  ");tree.write(world_path,encoding="utf-8",xml_declaration=True)
    plan={k:copy.deepcopy(v) for k,v in source.items() if k!="identity"}
    plan.update(annotation_mode="full_2d",label_mode="visual-instance",hierarchy_mode="top-level-equipment",calibration_views=views,pilot_views=[],diagnostic_only=True,diagnostic_allow_expected_absence=True,diagnostic_require_expected_presence=False,diagnostic_require_no_targets=True,training_admitted=False,promotable=False,automatic_training=False,diagnostic_purpose="isolated_hard_negative_v2",lighting_id=light_id,material_id="material_original",removed_target_models=sorted(removed),source_layout_id="complex-canonical-v1",derived_layout_id="complex-isolated-hard-negative-v2")
    plan["files"]["world.sdf"]=file_sha256(world_path)
    return write_record(folder/"plan.json",plan)


def prepare():
    path=BASE/"matrix.json"
    if path.exists():return json.loads(path.read_text())
    source=read_record(SOURCE);config=json.loads(CONFIG.read_text());poses=negative_candidates(source,config);runs=[];entries=[]
    for light_id in LIGHTS:
        run_id=f"hard-negative-{light_id}";views=[]
        for pose in poses:
            row=copy.deepcopy(pose);row.update(pose_id=pose["view_id"],derivation_group=f"negative-v2:{pose['view_id']}",lighting_id=light_id,material_id="material_original",data_role="new_training_candidate");views.append(row)
        plan=make_plan(source,views,run_id,light_id)
        runs.append({"run_id":run_id,"plan_path":str(BASE/"runs"/run_id/"plan/plan.json"),"plan_identity":plan["identity"],"frame_count":24,"lighting_id":light_id,"world_sha256":plan["files"]["world.sdf"]})
        for row in views:
            entries.append({"candidate_id":object_sha256({"run":run_id,"pose":row["pose_id"]}),"data_role":"new_training_candidate","map_layout_id":"complex-isolated-hard-negative-v2","asset_family_ids":["ordinary-cabinet-v1","control-building-v1","pole-v1","perimeter-and-ground-v1"],"equipment_instance_id":None,"hard_negative_subject":row["object_id"],"hard_negative_type":row["hard_negative_type"],"recording_group":run_id,"derivation_group":row["derivation_group"],"pose_id":row["pose_id"],"material_id":"material_original","lighting_id":light_id,"annotation_mode":"full_2d","image_sha256":None,"label_sha256":None,"world_sha256":plan["files"]["world.sdf"],"sensor_sha256":plan["files"]["sensor_source.sdf"],"instance_present":False,"visibility_status":"not_applicable","truncation_status":"not_applicable","capture_status":"planned","review_decision":"pending","review_reason":None,"review_nature":"pending","reviewed_at":None,"exact_duplicate_of":None,"pixel_duplicate_of":None,"lineage_relations":[row["derivation_group"],"supersedes:visual-augmentation-240-v1:hard-negative"],"dedup_status":"designed_variant","admission_status":"pending","failure_reason":None})
    matrix={"schema_version":1,"status":"frozen_pending_pilot","dataset_version":"hard-negative-isolated-v2","supersedes_scope":"visual-augmentation-240-v1 hard-negative subset only","positive_subsets_unchanged":True,"counts":{"poses":24,"lights_per_pose":2,"frames":48},"runs":runs,"removed_target_categories":sorted(TARGETS),"training_admitted":False,"promotable":False};matrix["identity"]=object_sha256(matrix)
    ledger={"schema_version":1,"dataset_version":"hard-negative-isolated-v2","status":"frozen_pending_capture","entries":entries,"training_admitted":False,"promotable":False};ledger["identity"]=object_sha256(ledger)
    BASE.mkdir(parents=True,exist_ok=True);write_json(path,matrix);write_json(BASE/"intake-ledger.json",ledger);return matrix


if __name__=="__main__":print(json.dumps(prepare(),indent=2))
