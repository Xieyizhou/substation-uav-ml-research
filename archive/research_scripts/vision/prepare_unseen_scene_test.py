"""Seal an Extreme-layout test plan before augmentation training begins."""
import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.expansion import CLASSES, background_candidates, target_candidates
from src.vision.canonical.gates import validate_view_pose
from src.vision.canonical.plan import materialize,read_record,write_record
from scripts.vision.prepare_visual_augmentation_batch import BASE,spread


def prepare():
    seal=BASE/"unseen-scene-test-plan.json"
    if seal.exists():return json.loads(seal.read_text())
    source_dir=BASE/"unseen-extreme-source-v2"
    source=read_record(source_dir/"plan.json") if source_dir.exists() else materialize("extreme",source_dir,label_mode="visual-instance",hierarchy_mode="top-level-equipment",instance_collision_strategy="linear-probe")
    config=json.loads((source_dir/"obstacles.json").read_text()); origin=config["gazebo_world_origin_m"][:2];size=(config["width"],config["height"])
    candidates=target_candidates("extreme",source["objects"],origin,size,241); positives=[]
    for category in CLASSES:
        legal=[]
        for row in candidates:
            if row["category"]!=category:continue
            try:validate_view_pose(row,config)
            except ValueError:continue
            legal.append(row)
        positives.extend(spread(legal,4))
    negatives=[]
    for row in background_candidates("extreme",source["objects"],origin,size,251):
        try:validate_view_pose(row,config)
        except ValueError:continue
        negatives.append(row)
        if len(negatives)==8:break
    views=[]
    for row in positives+negatives:
        item=copy.deepcopy(row);item["data_role"]="frozen_unseen_scene_test";item["derivation_group"]=f"extreme:{row['view_id']}";views.append(item)
    plan={k:copy.deepcopy(v) for k,v in source.items() if k!="identity"}
    plan.update(annotation_mode="full_2d",calibration_views=views,pilot_views=[],diagnostic_only=True,diagnostic_require_expected_presence=True,diagnostic_allow_expected_absence=False,training_admitted=False,promotable=False,automatic_training=False,diagnostic_purpose="sealed_unseen_extreme_scene_test")
    plan["files"]["world.sdf"]=file_sha256(source_dir/"world.sdf");written=write_record(BASE/"unseen-scene-plan.json",plan)
    record={"schema_version":1,"status":"sealed_not_evaluated","data_role":"frozen_unseen_scene_test","map_layout_id":"extreme-canonical-v1","asset_family_ids":["extreme-transformers-v1","extreme-switchgear-v1","extreme-capacitors-v1","extreme-reactors-v1"],"plan_path":str(BASE/"unseen-scene-plan.json"),"plan_identity":written["identity"],"positive_frames":16,"planned_negative_frames":8,"instance_collision_resolutions":source.get("instance_collision_resolutions",[]),"freeze_rule":"Do not capture/evaluate until development arms are complete and a candidate is selected without using this test set.","training_admitted":False,"promotable":False};record["identity"]=object_sha256(record);write_json(seal,record);return record


if __name__=="__main__":print(json.dumps(prepare(),indent=2))
