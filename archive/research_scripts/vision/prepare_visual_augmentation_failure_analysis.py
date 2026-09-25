"""Freeze failure-analysis definitions before computing diagnostic summaries."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json

BASE=ROOT/"data/research/ml_training_recovery_v1"
OUT=BASE/"visual-augmentation-failure-analysis-v1"
SOURCES=[
    BASE/"visual-augmentation-240-v2/frozen-intake-ledger.json",
    BASE/"visual-augmentation-training-v2/protocol.json",
    BASE/"visual-augmentation-composition-v1/protocol.json",
    BASE/"visual-augmentation-composition-v1/development-evaluation.json",
    BASE/"visual-augmentation-composition-v1/completion.json",
    BASE/"hard-negative-isolated-v2/semantic-review.json",
]


def prepare():
    path=OUT/"protocol.json"
    if path.exists():
        result=json.loads(path.read_text())
        for source,digest in result["input_hashes"].items():
            if file_sha256(Path(source))!=digest: raise ValueError("Frozen analysis input changed")
        return result
    completion=json.loads(SOURCES[4].read_text())
    if completion["status"]!="development_complete_no_candidate" or completion["candidate_family"] is not None:
        raise ValueError("Failure analysis requires a closed development round without a candidate")
    result={"schema_version":1,"status":"frozen_before_analysis","analysis_id":"visual-augmentation-failure-analysis-v1",
        "scope":{"appearance_members":"training protocol rows where subset=appearance","negative_members":"48 hash-bound reviewed isolated negatives",
            "model_families":["E","F"],"seeds":[7,17,27],"unseen_scene":"excluded_and_sealed"},
        "target_bbox_area_bins":[{"name":"tiny","lower_inclusive":0.0,"upper_exclusive":0.01},
            {"name":"small","lower_inclusive":0.01,"upper_exclusive":0.04},
            {"name":"medium","lower_inclusive":0.04,"upper_exclusive":0.16},
            {"name":"large","lower_inclusive":0.16,"upper_exclusive":None}],
        "appearance_metrics":["target_bbox_area_fraction","target_bbox_width_px","target_bbox_height_px","target_crop_luma_mean",
            "target_crop_luma_stddev","target_crop_saturation_mean","category_material_lighting_counts","derivation_group_variant_count",
            "target_bbox_max_delta_px_within_group","label_hash_count_within_group"],
        "material_palette":{"mat_cool_gray":[0.35,0.37,0.40,1.0],"mat_warm_oxide":[0.43,0.31,0.23,1.0],
            "mat_desaturated_green":[0.28,0.38,0.32,1.0],"paired_regression_neutral":[0.35,0.35,0.35,1.0]},
        "material_assignment_rule":"The run material id is a palette rotation; report the actual per-category material, not the run id.",
        "negative_subject_families":{"cabinet_center":"ordinary_cabinet","control_building":"control_building",
            "pole_a":"utility_pole","pole_b":"utility_pole","pole_c":"utility_pole","pole_d":"utility_pole"},
        "negative_metrics":["frames","frames_with_predictions","frame_false_positive_rate","prediction_count","predicted_class_counts",
            "confidence_mean","confidence_max"],
        "highlight_rule":"For each family, identify the seed with maximum isolated-negative frame false-positive rate; retain results for every seed.",
        "interpretation_limits":["Training members and viewed development frames are diagnostic, not independent validation.",
            "Pixel statistics describe rendered target crops and do not establish causal features.",
            "No protected labels, unseen-scene members, threshold tuning, training, or promotion."],
        "input_hashes":{str(p):file_sha256(p) for p in SOURCES},"unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);OUT.mkdir(parents=True,exist_ok=False);write_json(path,result);return result


if __name__=="__main__": print(json.dumps(prepare(),ensure_ascii=False,indent=2))
