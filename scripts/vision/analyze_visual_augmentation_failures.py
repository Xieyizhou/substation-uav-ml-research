"""Analyze appearance coverage/geometry and isolated-negative confusions."""
import json
import statistics
import sys
from collections import Counter,defaultdict
from pathlib import Path

from PIL import Image,ImageStat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_visual_augmentation_failure_analysis import OUT,prepare

BASE=ROOT/"data/research/ml_training_recovery_v1"
TRAINING=BASE/"visual-augmentation-training-v2/protocol.json"
LEDGER=BASE/"visual-augmentation-240-v2/frozen-intake-ledger.json"
EVALUATION=BASE/"visual-augmentation-composition-v1/development-evaluation.json"
NEGATIVE_REVIEW=BASE/"hard-negative-isolated-v2/semantic-review.json"
MATERIALS=["mat_cool_gray","mat_warm_oxide","mat_desaturated_green"]
CLASSES=["transformer","switchgear","capacitor_bank","reactor"]


def actual_material(run_material,category):
    shift=MATERIALS.index(run_material);index=CLASSES.index(category)
    return MATERIALS[(index+shift)%len(MATERIALS)]


def area_bin(value,bins):
    for row in bins:
        if value>=row["lower_inclusive"] and (row["upper_exclusive"] is None or value<row["upper_exclusive"]): return row["name"]
    raise ValueError("Target area was not binned")


def target_from_receipt(image_path,image_sha):
    receipt_path=image_path.parents[1]/"collection-receipt.json";receipt=json.loads(receipt_path.read_text())
    view=next(v for v in receipt["views"] if v["image_sha256"]==image_sha)
    mapping=receipt["collection_checks"]["instance_mapping"]
    labels=[int(label) for label,row in mapping.items() if row["object_id"]==view["expected_object_id"]]
    if len(labels)!=1: raise ValueError("Planned instance mapping is ambiguous")
    raw=next(row for row in view["raw_truth"]["annotatedBox"] if row["label"]==labels[0])["box"]
    lo=raw.get("minCorner",{});hi=raw.get("maxCorner",{})
    bbox=[float(lo.get("x",0)),float(lo.get("y",0)),float(hi.get("x",0)),float(hi.get("y",0))]
    return view,bbox,receipt_path


def summarize_numbers(values):
    return {"min":min(values),"median":statistics.median(values),"mean":statistics.mean(values),"max":max(values)}


def main():
    protocol=prepare();training=json.loads(TRAINING.read_text());ledger=json.loads(LEDGER.read_text())
    by_candidate={row["candidate_id"]:row for row in ledger["entries"]};appearance=[];receipt_hashes={}
    for member in (row for row in training["pool_rows"] if row["subset"]=="appearance"):
        ledger_row=by_candidate[member["source_candidate_id"]];image_path=Path(member["image_path"])
        if file_sha256(image_path)!=member["image_sha256"] or file_sha256(Path(member["label_path"]))!=member["label_sha256"]:
            raise ValueError("Appearance member is stale")
        source_image_path=Path(ledger_row["image_path"])
        if file_sha256(source_image_path)!=ledger_row["image_sha256"]: raise ValueError("Source candidate image is stale")
        view,bbox,receipt_path=target_from_receipt(source_image_path,ledger_row["image_sha256"]);receipt_hashes[str(receipt_path)]=file_sha256(receipt_path)
        with Image.open(image_path) as image:
            width,height=image.size;x1,y1,x2,y2=bbox;crop=image.convert("RGB").crop((round(x1),round(y1),round(x2),round(y2)))
            rgb=ImageStat.Stat(crop);gray=ImageStat.Stat(crop.convert("L"));sat=ImageStat.Stat(crop.convert("HSV"))
        area=max(0,x2-x1)*max(0,y2-y1)/(width*height)
        appearance.append({"member_id":member["member_id"],"image_sha256":member["image_sha256"],"label_sha256":member["label_sha256"],
            "derivation_group":ledger_row["derivation_group"],"pose_id":ledger_row["pose_id"],"category":view["expected_category"],
            "object_id":view["expected_object_id"],"run_material_id":ledger_row["material_id"],
            "actual_material_id":actual_material(ledger_row["material_id"],view["expected_category"]),"lighting_id":ledger_row["lighting_id"],
            "target_bbox_xyxy":bbox,"target_bbox_width_px":x2-x1,"target_bbox_height_px":y2-y1,"target_bbox_area_fraction":area,
            "target_bbox_area_bin":area_bin(area,protocol["target_bbox_area_bins"]),"target_crop_rgb_mean":rgb.mean,
            "target_crop_luma_mean":gray.mean[0],"target_crop_luma_stddev":gray.stddev[0],"target_crop_saturation_mean":sat.mean[1]/255})
    groups=defaultdict(list)
    for row in appearance: groups[row["derivation_group"]].append(row)
    lineage=[]
    for group,rows in sorted(groups.items()):
        reference=rows[0]["target_bbox_xyxy"]
        lineage.append({"derivation_group":group,"variant_count":len(rows),"label_hash_count":len({r["label_sha256"] for r in rows}),
            "target_bbox_max_delta_px":max(abs(a-b) for row in rows for a,b in zip(row["target_bbox_xyxy"],reference))})
    dimensions={}
    for category in CLASSES:
        rows=[r for r in appearance if r["category"]==category]
        dimensions[category]={"frames":len(rows),"independent_poses":len({r["pose_id"] for r in rows}),
            "by_actual_material":dict(sorted(Counter(r["actual_material_id"] for r in rows).items())),
            "by_lighting":dict(sorted(Counter(r["lighting_id"] for r in rows).items())),
            "area_fraction":summarize_numbers([r["target_bbox_area_fraction"] for r in rows]),
            "area_bins":dict(sorted(Counter(r["target_bbox_area_bin"] for r in rows).items()))}
    render={}
    for material in MATERIALS:
        render[material]={}
        for light in ("light_normal","light_cool_low"):
            rows=[r for r in appearance if r["actual_material_id"]==material and r["lighting_id"]==light]
            render[material][light]={"frames":len(rows),"luma_mean":statistics.mean(r["target_crop_luma_mean"] for r in rows),
                "luma_stddev_mean":statistics.mean(r["target_crop_luma_stddev"] for r in rows),
                "saturation_mean":statistics.mean(r["target_crop_saturation_mean"] for r in rows)}
    review=json.loads(NEGATIVE_REVIEW.read_text());evaluation=json.loads(EVALUATION.read_text())
    review_by_key={(r["view_id"],r["variant"]):r for r in review["frames"]};negative={}
    families=protocol["negative_subject_families"]
    for arm in "EF":
        per_seed={}
        for seed in (7,17,27):
            rows=evaluation["results"][f"{arm}_{seed}"]["negative_rows"];joined=[]
            for row in rows:
                meta=review_by_key[(row["view_id"],row["variant"])];family=families[meta["subject"]]
                joined.append((row,meta,family))
            by_family={}
            for family in sorted(set(families.values())):
                selected=[x for x in joined if x[2]==family];predictions=[p for row,_,_ in selected for p in row["predictions"]]
                confidences=[p["confidence"] for p in predictions]
                by_family[family]={"frames":len(selected),"frames_with_predictions":sum(row["frame_has_prediction"] for row,_,_ in selected),
                    "frame_false_positive_rate":sum(row["frame_has_prediction"] for row,_,_ in selected)/len(selected),
                    "prediction_count":len(predictions),"predicted_class_counts":dict(sorted(Counter(p["class_name"] for p in predictions).items())),
                    "confidence_mean":statistics.mean(confidences) if confidences else None,"confidence_max":max(confidences) if confidences else None}
            all_predictions=[p for row,_,_ in joined for p in row["predictions"]]
            per_seed[str(seed)]={"summary":evaluation["results"][f"{arm}_{seed}"]["negative_summary"],"by_subject_family":by_family,
                "predicted_class_counts":dict(sorted(Counter(p["class_name"] for p in all_predictions).items()))}
        worst=max(per_seed,key=lambda seed:per_seed[seed]["summary"]["frame_false_positive_rate"])
        negative[arm]={"worst_fpr_seed":int(worst),"seeds":per_seed}
    negative_coverage={family:sum(families[r["subject"]]==family for r in review["frames"]) for family in sorted(set(families.values()))}
    result={"schema_version":1,"status":"analysis_complete","protocol_identity":protocol["identity"],"appearance":{"frames":len(appearance),
        "independent_pose_groups":len(groups),"category_summary":dimensions,"render_summary":render,
        "lineage_summary":{"groups":len(lineage),"groups_with_six_variants":sum(r["variant_count"]==6 for r in lineage),
            "groups_with_one_label_hash":sum(r["label_hash_count"]==1 for r in lineage),"maximum_target_bbox_delta_px":max(r["target_bbox_max_delta_px"] for r in lineage)},
        "rows":appearance,"lineage_rows":lineage},"negative":{"coverage_frames":negative_coverage,"families":negative},
        "input_hashes":{str(Path(__file__)):file_sha256(Path(__file__)),str(OUT/"protocol.json"):file_sha256(OUT/"protocol.json"),
            str(TRAINING):file_sha256(TRAINING),str(LEDGER):file_sha256(LEDGER),str(EVALUATION):file_sha256(EVALUATION),
            str(NEGATIVE_REVIEW):file_sha256(NEGATIVE_REVIEW),**receipt_hashes},"unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False,"limits":protocol["interpretation_limits"]}
    result["identity"]=object_sha256(result);write_json(OUT/"analysis.json",result)
    print(json.dumps({"status":result["status"],"identity":result["identity"],"appearance_frames":result["appearance"]["frames"],
        "independent_pose_groups":result["appearance"]["independent_pose_groups"],"category_summary":result["appearance"]["category_summary"],
        "render_summary":result["appearance"]["render_summary"],"lineage_summary":result["appearance"]["lineage_summary"],
        "negative_coverage_frames":result["negative"]["coverage_frames"],
        "worst_fpr_seeds":{arm:result["negative"]["families"][arm]["worst_fpr_seed"] for arm in "EF"},
        "unseen_scene_status":result["unseen_scene_status"],"training_admitted":False,"promotable":False},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
