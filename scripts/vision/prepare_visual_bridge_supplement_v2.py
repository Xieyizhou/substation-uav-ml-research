"""Supersede the rejected v1 bridge matrix with persisted full_2d worlds."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.gates import annotation_mode_from_world
from src.vision.canonical.plan import read_record,write_record
from scripts.vision.prepare_paired_visual_factors import mutate,assert_allowed_world_diff
import scripts.vision.prepare_visual_bridge_supplement as v1

BASE=ROOT/"data/research/ml_training_recovery_v1/visual-bridge-supplement-v2"
REJECTED=ROOT/"data/research/ml_training_recovery_v1/visual-bridge-supplement-v1/matrix.json"


def make_plan(source,views,variant):
    folder=BASE/"runs"/variant/"plan";folder.mkdir(parents=True,exist_ok=False)
    for name,digest in source["files"].items():
        src=v1.SOURCE.parent/name
        if file_sha256(src)!=digest: raise ValueError("Canonical source artifact changed")
        (folder/name).write_bytes(src.read_bytes())
    baseline=folder/"baseline-full2d.sdf"
    tree=ET.parse(folder/"world.sdf");sensors=[row for row in tree.iter("sensor") if row.get("type")=="boundingbox_camera"]
    if len(sensors)!=1: raise ValueError("Expected one bounding-box camera")
    box_type=sensors[0].find("camera/box_type")
    if box_type is None: raise ValueError("Missing camera box_type")
    box_type.text="full_2d";ET.indent(tree,space="  ");tree.write(baseline,encoding="utf-8",xml_declaration=True)
    tree=ET.parse(baseline);mapped="original" if variant=="original" else "material" if variant=="neutral_bridge" else "background"
    changed=mutate(tree,mapped,source["objects"]);ET.indent(tree,space="  ");tree.write(folder/"world.sdf",encoding="utf-8",xml_declaration=True)
    if mapped!="original": assert_allowed_world_diff(baseline,folder/"world.sdf",mapped)
    if annotation_mode_from_world(folder/"world.sdf")!="full_2d": raise ValueError("Persisted world mode verification failed")
    baseline.unlink()
    record={k:copy.deepcopy(value) for k,value in source.items() if k!="identity"};variant_views=[]
    for base in views:
        row=copy.deepcopy(base);row["view_id"]=object_sha256({"bridge_round":v1.ROUND,"source_view_id":base["source_view_id"],"variant":variant,"revision":2})
        row.update(variant=variant,pair_id=object_sha256({"bridge_round":v1.ROUND,"source_view_id":base["source_view_id"]}),
            data_role="new_training_candidate",training_admitted=False,promotable=False);variant_views.append(row)
    record.update(annotation_mode="full_2d",label_mode="visual-instance",hierarchy_mode="top-level-equipment",calibration_views=variant_views,
        pilot_views=[],diagnostic_only=True,diagnostic_require_expected_presence=True,training_admitted=False,promotable=False,
        automatic_training=False,diagnostic_purpose="visual_bridge_supplement",variant=variant,bridge_round=v1.ROUND,
        check_correction="persisted_full_2d_before_launch",allowed_world_changes=changed)
    record["files"]["world.sdf"]=file_sha256(folder/"world.sdf");return write_record(folder/"plan.json",record),folder


def prepare():
    path=BASE/"matrix.json"
    if path.exists(): return json.loads(path.read_text())
    rejected=json.loads(REJECTED.read_text());source=read_record(v1.SOURCE);config=json.loads(v1.CONFIG.read_text())
    background=json.loads(v1.BACKGROUND.read_text());views,known_count=v1.choose(source,config);runs=[];pilot=[]
    for variant in v1.VARIANTS:
        plan,folder=make_plan(source,views,variant)
        old_base=v1.BASE;v1.BASE=BASE
        try: pilot_plan,pilot_folder=v1.make_pilot(plan,folder)
        finally: v1.BASE=old_base
        runs.append({"variant":variant,"frame_count":16,"plan_path":str(folder/"plan.json"),"plan_identity":plan["identity"],"world_sha256":plan["files"]["world.sdf"],"actual_annotation_mode":"full_2d"})
        pilot.append({"variant":variant,"frame_count":2,"plan_path":str(pilot_folder/"plan.json"),"plan_identity":pilot_plan["identity"],"world_sha256":pilot_plan["files"]["world.sdf"],"actual_annotation_mode":"full_2d"})
    negatives=v1.negative_redesign(background)
    result={"schema_version":2,"status":"frozen_pending_pilot","dataset_version":"visual-bridge-supplement-v2",
        "supersedes":str(REJECTED),"superseded_identity":rejected["identity"],"supersession_reason":"v1 pilot was rejected before Gazebo launch because declared full_2d did not match persisted visible_2d world configuration; no frames were captured.",
        "positive":{"frame_count":48,"independent_pose_groups":16,"categories":{"capacitor_bank":24,"reactor":24},"variants":list(v1.VARIANTS),
            "runs":runs,"selection":{"round":v1.ROUND,"rule":"eight farthest legal unblocked new poses per priority class","known_and_prior_pose_ids_excluded":known_count}},
        "positive_pilot":{"status":"frozen_pending_capture","frame_count":6,"independent_pose_groups":2,"runs":pilot},
        "negative_redesign":{"status":"frozen_pending_plan_materialization_and_source_validation","frame_count":24,"independent_pose_groups":12,
            "family_frame_counts":dict(sorted(Counter(row["subject_family"] for row in negatives).items())),"rows":negatives},
        "input_hashes":{str(REJECTED):file_sha256(REJECTED),str(v1.SOURCE):file_sha256(v1.SOURCE),str(v1.CONFIG):file_sha256(v1.CONFIG),
            str(v1.FROZEN):file_sha256(v1.FROZEN),str(v1.FAILURE):file_sha256(v1.FAILURE),str(v1.BACKGROUND):file_sha256(v1.BACKGROUND),
            str(Path(__file__)):file_sha256(Path(__file__))},"constraints":rejected["constraints"],"unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);write_json(path,result);return result


if __name__=="__main__":
    result=prepare();print(json.dumps({"status":result["status"],"identity":result["identity"],"superseded_identity":result["superseded_identity"],
        "positive_frames":result["positive"]["frame_count"],"pilot_frames":result["positive_pilot"]["frame_count"],
        "negative_family_counts":result["negative_redesign"]["family_frame_counts"],"unseen_scene_status":result["unseen_scene_status"]},indent=2))
