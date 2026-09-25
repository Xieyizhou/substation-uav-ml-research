"""Freeze preservation-heavy compositions and acceptance rules before training."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json

BASE=ROOT/"data/research/ml_training_recovery_v1"
SOURCE=BASE/"visual-augmentation-training-v2/protocol.json"
PRIOR_EVAL=BASE/"visual-augmentation-training-v2/development-evaluation.json"
OUT=BASE/"visual-augmentation-composition-v1"
SEEDS=(7,17,27); EPOCHS=10; SLOTS=60; BATCH=6
QUOTAS={"E":{"base":180,"regular":150,"appearance":150,"negative":120},"F":{"base":162,"regular":132,"appearance":186,"negative":120}}


def cycle(rows,count,rng):
    result=[]
    while len(result)<count:
        values=list(rows);rng.shuffle(values);result.extend(values)
    return result[:count]


def schedule(rows,quotas,seed):
    rng=random.Random(seed);by_subset={name:[r["member_id"] for r in rows if r["subset"]==name] for name in quotas};draws=[]
    for subset,count in quotas.items(): draws.extend(cycle(by_subset[subset],count,rng))
    rng.shuffle(draws)
    if len(draws)!=600 or any(not set(by_subset[s])<=set(draws) for s in quotas): raise ValueError("Composition schedule is incomplete")
    return [draws[i*SLOTS:(i+1)*SLOTS] for i in range(EPOCHS)]


def prepare():
    path=OUT/"protocol.json"
    if path.exists():
        result=json.loads(path.read_text())
        for p,h in result["input_hashes"].items():
            if file_sha256(Path(p))!=h: raise ValueError("Frozen composition input changed")
        return result
    source=json.loads(SOURCE.read_text());prior=json.loads(PRIOR_EVAL.read_text())
    if source["status"]!="frozen_before_training" or prior["status"]!="development_evaluation_complete": raise ValueError("Prior development cycle is incomplete")
    rows=source["pool_rows"]; membership=source["membership"]["D"]
    OUT.mkdir(parents=True,exist_ok=False)
    source_list=Path(membership["path"]);dataset_yamls={};schedules={};exposures={}
    for arm,quotas in QUOTAS.items():
        list_path=OUT/f"arm-{arm}.txt";list_path.write_text(source_list.read_text())
        yaml_path=OUT/f"dataset-{arm}.yaml";yaml_path.write_text(f"path: {OUT}\ntrain: {list_path}\nval: {list_path}\nnames: [transformer, switchgear, capacitor_bank, reactor]\n")
        dataset_yamls[arm]={"path":str(yaml_path),"sha256":file_sha256(yaml_path)};schedules[arm]={};exposures[arm]={}
        by_id={r["member_id"]:r for r in rows}
        for seed in SEEDS:
            value=schedule(rows,quotas,seed);schedules[arm][str(seed)]=value;draws=[x for epoch in value for x in epoch]
            exposures[arm][str(seed)]={"draws":600,"unique_frames":len(set(draws)),"by_subset":dict(sorted(Counter(by_id[x]["subset"] for x in draws).items()))}
    policy={"aggregation":"unweighted mean across seeds 7/17/27; no best-seed selection","required_all":[
        {"metric":"original.planned_instance_hit_rate.mean","op":">=","value":0.8333333333333334},
        {"metric":"original.planned_instance_hit_rate.min_seed","op":">=","value":0.75},
        {"metric":"material.planned_instance_hit_rate.mean","op":">=","value":0.5},
        {"metric":"material.planned_instance_hit_rate.min_seed","op":">=","value":0.3333333333333333},
        {"metric":"lighting.planned_instance_hit_rate.mean","op":">=","value":0.6},
        {"metric":"background.planned_instance_hit_rate.mean","op":">=","value":0.6},
        {"metric":"no_target.frame_false_positive_rate.mean","op":"<=","value":0.1},
        {"metric":"no_target.frame_false_positive_rate.max_seed","op":"<=","value":0.2}],
        "selection_order":"Select the E family if it passes every rule; otherwise select F only if it passes every rule; otherwise select none. Keep all three seeds as the candidate family.",
        "rationale":"One-hit granularity is 1/12 on paired variants. Original mean may lose at most one hit versus A's 11/12 mean; no seed may fall below 9/12. Material and lighting require directional gains, while isolated-negative FPR must remain at most 10% mean and 20% for every seed."}
    controls=dict(source["controls"]);controls.update(epochs=EPOCHS,slots_per_epoch=SLOTS,batch=BATCH,optimizer_steps=100)
    result={"schema_version":1,"status":"frozen_before_training","experiment":"visual-augmentation-composition-v1","arms":{"E":"preservation-heavy","F":"balanced-appearance"},"quotas":QUOTAS,"membership":membership,"pool_rows":rows,"dataset_yamls":dataset_yamls,"schedules":schedules,"exposures":exposures,"seeds":list(SEEDS),"controls":controls,"acceptance_policy":policy,"prior_development_evaluation_identity":prior["identity"],"input_hashes":{str(SOURCE):file_sha256(SOURCE),str(PRIOR_EVAL):file_sha256(PRIOR_EVAL),str(source_list):file_sha256(source_list),controls["initial_weights"]:file_sha256(Path(controls["initial_weights"]))},"unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);write_json(path,result);return result


if __name__=="__main__":
    result=prepare();print(json.dumps({"status":result["status"],"identity":result["identity"],"quotas":result["quotas"],"exposures":result["exposures"],"acceptance_policy":result["acceptance_policy"]},indent=2))
