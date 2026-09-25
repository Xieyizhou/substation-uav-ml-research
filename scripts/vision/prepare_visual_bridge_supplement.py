"""Freeze a small positive bridge matrix and a balanced negative redesign."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.expansion import target_candidates
from src.vision.canonical.gates import validate_view_pose
from src.vision.canonical.occlusion import view_blockers
from src.vision.canonical.plan import read_record,write_record
from scripts.vision.prepare_paired_visual_factors import mutate,assert_allowed_world_diff
import xml.etree.ElementTree as ET

BASE=ROOT/"data/research/ml_training_recovery_v1/visual-bridge-supplement-v1"
SOURCE=ROOT/"data/research/canonical_views_v1/complex-occlusion-filtered-plan-v1/plan.json"
CONFIG=ROOT/"config/maps/substation_complex.json"
FROZEN=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-240-v2/frozen-intake-ledger.json"
FAILURE=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-failure-analysis-v1/analysis.json"
BACKGROUND=ROOT/"data/research/background_calibration_v1/manifest.json"
ROUND=271
VARIANTS=("original","neutral_bridge","background_bridge")


def known_view_ids():
    known=set()
    for path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(path): continue
        try: payload=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError): continue
        known.update(row.get("view_id") for row in payload.get("views",[]) if row.get("view_id"))
    return known


def choose(source,config):
    prior={row["pose_id"] for row in json.loads(FROZEN.read_text())["entries"] if row["pose_id"]}
    known=known_view_ids()|prior;rows=target_candidates("complex",source["objects"],config["gazebo_world_origin_m"][:2],(config["width"],config["height"]),ROUND)
    selected=[]
    for category in ("capacitor_bank","reactor"):
        legal=[]
        for row in rows:
            if row["category"]!=category or row["view_id"] in known: continue
            try: validate_view_pose(row,config)
            except ValueError: continue
            if not view_blockers(row,source["objects"]): legal.append(row)
        legal=sorted(legal,key=lambda row:(-row["distance"],row["bearing"],row["height"],row["view_id"]))
        if len(legal)<8: raise ValueError(f"Need eight legal new far poses for {category}")
        for rank,row in enumerate(legal[:8]):
            item=copy.deepcopy(row);item.update(source_view_id=row["view_id"],pose_id=row["view_id"],distance_priority_rank=rank+1,
                derivation_group=f"bridge-pose:{row['view_id']}")
            selected.append(item)
    return selected,len(known)


def make_plan(source,views,variant):
    folder=BASE/"runs"/variant/"plan";folder.mkdir(parents=True,exist_ok=False)
    for name,digest in source["files"].items():
        src=SOURCE.parent/name
        if file_sha256(src)!=digest: raise ValueError("Canonical source artifact changed")
        (folder/name).write_bytes(src.read_bytes())
    original=folder/"world-original.sdf";(folder/"world.sdf").replace(original)
    tree=ET.parse(original)
    mapped="original" if variant=="original" else "material" if variant=="neutral_bridge" else "background"
    changed=mutate(tree,mapped,source["objects"]);ET.indent(tree,space="  ");tree.write(folder/"world.sdf",encoding="utf-8",xml_declaration=True)
    if mapped!="original": assert_allowed_world_diff(original,folder/"world.sdf",mapped)
    original.unlink()
    record={k:copy.deepcopy(v) for k,v in source.items() if k!="identity"}
    variant_views=[]
    for base in views:
        row=copy.deepcopy(base);row["view_id"]=object_sha256({"bridge_round":ROUND,"source_view_id":base["source_view_id"],"variant":variant})
        row.update(variant=variant,pair_id=object_sha256({"bridge_round":ROUND,"source_view_id":base["source_view_id"]}),
            data_role="new_training_candidate",training_admitted=False,promotable=False);variant_views.append(row)
    record.update(annotation_mode="full_2d",label_mode="visual-instance",hierarchy_mode="top-level-equipment",calibration_views=variant_views,
        pilot_views=[],diagnostic_only=True,diagnostic_require_expected_presence=True,training_admitted=False,promotable=False,
        automatic_training=False,diagnostic_purpose="visual_bridge_supplement",variant=variant,bridge_round=ROUND,allowed_world_changes=changed)
    record["files"]["world.sdf"]=file_sha256(folder/"world.sdf")
    return write_record(folder/"plan.json",record),folder


def make_pilot(plan,folder):
    target=BASE/"pilot-v1"/"runs"/plan["variant"]/"plan";target.mkdir(parents=True,exist_ok=False)
    for path in folder.iterdir():
        if path.name!="plan.json": (target/path.name).write_bytes(path.read_bytes())
    chosen=[]
    for category in ("capacitor_bank","reactor"):
        chosen.append(next(row for row in plan["calibration_views"] if row["category"]==category))
    record={k:copy.deepcopy(v) for k,v in plan.items() if k!="identity"};record["calibration_views"]=chosen
    record["parent_plan_identity"]=plan["identity"];return write_record(target/"plan.json",record),target


def negative_redesign(background):
    sources={"ordinary_cabinet":"complex-canonical-v1","control_building":"complex-canonical-v1","utility_pole":"complex-canonical-v1",
        "fence":"bg_fence_yard_v1","masonry_wall":"bg_masonry_lane_v1","pipe_rack_gantry":"bg_pipe_rack_v1"}
    scene_by_id={row["scene_id"]:row for row in background["scenes"]};rows=[]
    for family,source in sources.items():
        source_hash=file_sha256(SOURCE) if source=="complex-canonical-v1" else scene_by_id[source]["files"]["world.sdf"]["sha256"]
        for pose_index in range(2):
            group=f"negative-bridge:{family}:{pose_index}"
            for light in ("light_normal","light_cool_low"):
                rows.append({"candidate_id":object_sha256({"group":group,"light":light}),"subject_family":family,"source_scene":source,
                    "source_sha256":source_hash,"pose_slot":pose_index,"lighting_id":light,"derivation_group":group,
                    "capture_status":"planned_pending_source_live_validation" if source!="complex-canonical-v1" else "planned",
                    "empty_truth_required":True,"visual_target_exclusion_review_required":True,"training_admitted":False,"promotable":False})
    return rows


def prepare():
    path=BASE/"matrix.json"
    if path.exists(): return json.loads(path.read_text())
    failure=json.loads(FAILURE.read_text())
    if failure["status"]!="analysis_complete" or failure["unseen_scene_status"]!="sealed_not_evaluated": raise ValueError("Failure analysis is not closed")
    source=read_record(SOURCE);config=json.loads(CONFIG.read_text());background=json.loads(BACKGROUND.read_text())
    views,known_count=choose(source,config);runs=[];pilot=[]
    for variant in VARIANTS:
        plan,folder=make_plan(source,views,variant);pilot_plan,pilot_folder=make_pilot(plan,folder)
        runs.append({"variant":variant,"frame_count":16,"plan_path":str(folder/"plan.json"),"plan_identity":plan["identity"],
            "world_sha256":plan["files"]["world.sdf"]})
        pilot.append({"variant":variant,"frame_count":2,"plan_path":str(pilot_folder/"plan.json"),"plan_identity":pilot_plan["identity"],
            "world_sha256":pilot_plan["files"]["world.sdf"]})
    negatives=negative_redesign(background)
    result={"schema_version":1,"status":"frozen_pending_pilot","dataset_version":"visual-bridge-supplement-v1",
        "positive":{"frame_count":48,"independent_pose_groups":16,"categories":{"capacitor_bank":24,"reactor":24},
            "variants":list(VARIANTS),"runs":runs,"selection":{"round":ROUND,"rule":"eight farthest legal unblocked new poses per priority class",
                "known_and_prior_pose_ids_excluded":known_count}},
        "positive_pilot":{"status":"frozen_pending_capture","frame_count":6,"independent_pose_groups":2,"runs":pilot},
        "negative_redesign":{"status":"frozen_pending_plan_materialization_and_source_validation","frame_count":24,"independent_pose_groups":12,
            "family_frame_counts":dict(sorted(Counter(row["subject_family"] for row in negatives).items())),"rows":negatives},
        "input_hashes":{str(SOURCE):file_sha256(SOURCE),str(CONFIG):file_sha256(CONFIG),str(FROZEN):file_sha256(FROZEN),
            str(FAILURE):file_sha256(FAILURE),str(BACKGROUND):file_sha256(BACKGROUND),str(Path(__file__)):file_sha256(Path(__file__))},
        "constraints":["Positive variants share pose, target instance, geometry, sensor and label mapping.",
            "Custom negative sources require live synchronization and empty-truth validation before capture admission.",
            "Pilot review and deduplication must pass before remaining capture.","No unseen-scene access, training, threshold tuning or promotion."],
        "unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);write_json(path,result);return result


if __name__=="__main__":
    result=prepare();print(json.dumps({"status":result["status"],"identity":result["identity"],"positive":result["positive"],
        "positive_pilot":result["positive_pilot"],"negative_family_counts":result["negative_redesign"]["family_frame_counts"],
        "unseen_scene_status":result["unseen_scene_status"],"training_admitted":False,"promotable":False},ensure_ascii=False,indent=2))
