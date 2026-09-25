"""Freeze a 20-frame workflow pilot sampled from the already frozen 240 matrix."""
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256,write_json
from src.vision.canonical.plan import read_record,write_record
from scripts.vision.prepare_visual_augmentation_batch import BASE

PILOT=BASE/"pilot-v1"
RUNS=("appearance-mat_cool_gray-light_normal","appearance-mat_warm_oxide-light_cool_low","hard-negative-light_normal","hard-negative-light_cool_low","regular-positive")


def select(plan,run_id):
    rows=plan["calibration_views"]
    if run_id.startswith("hard-negative"):return rows[:4]
    selected=[]
    for category in ("capacitor_bank","switchgear"):
        selected.extend([r for r in rows if r["category"]==category][:2])
    return selected


def prepare():
    path=PILOT/"manifest.json"
    if path.exists():return json.loads(path.read_text())
    runs=[]
    for run_id in RUNS:
        source_dir=BASE/"runs"/run_id/"plan"; target=PILOT/"runs"/run_id/"plan"
        shutil.copytree(source_dir,target)
        source=read_record(source_dir/"plan.json"); plan={k:copy.deepcopy(v) for k,v in source.items() if k!="identity"}
        plan["calibration_views"]=select(plan,run_id);plan["pilot_parent_plan_identity"]=source["identity"];plan["diagnostic_purpose"]="visual_augmentation_workflow_pilot"
        (target/"plan.json").unlink();written=write_record(target/"plan.json",plan)
        runs.append({"run_id":run_id,"plan_path":str(target/"plan.json"),"plan_identity":written["identity"],"frame_count":len(plan["calibration_views"]),"require_no_targets":plan.get("diagnostic_require_no_targets",False)})
    manifest={"schema_version":1,"status":"frozen_pending_capture","purpose":"workflow pilot only; does not alter the frozen 240 candidate matrix","frame_count":20,"runs":runs,"training_admitted":False,"promotable":False};manifest["identity"]=object_sha256(manifest);write_json(path,manifest);return manifest


if __name__=="__main__":print(json.dumps(prepare(),indent=2))
