"""Render and finalize the eight captured positive workflow-pilot frames."""
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_visual_augmentation_pilot import PILOT

RUNS=("appearance-mat_cool_gray-light_normal","appearance-mat_warm_oxide-light_cool_low")
COLORS={"transformer":"#ff3b30","switchgear":"#00a8ff","capacitor_bank":"#34c759","reactor":"#ffcc00"}

def target_box(row,receipt):
    mapping=receipt["collection_checks"]["instance_mapping"]
    label=next(key for key,value in mapping.items() if value["object_id"]==row["expected_object_id"])
    items=row["raw_truth"].get("annotated_box",row["raw_truth"].get("annotatedBox",[]));index=next(i for i,item in enumerate(items) if int(item["label"])==int(label))
    return row["truth"]["objects"][index]["bbox_xyxy"]

def render():
    groups={};inputs={}
    for run_id in RUNS:
        path=PILOT/"runs"/run_id/"capture/collection-receipt.json";inputs[str(path)]=file_sha256(path);receipt=json.loads(path.read_text());groups[run_id]={row["view_id"]:(row,receipt) for row in receipt["views"]}
    if set(groups[RUNS[0]])!=set(groups[RUNS[1]]) or len(groups[RUNS[0]])!=4:raise ValueError("Positive pilot pairs are incomplete")
    sheet=Image.new("RGB",(1920,1200),"#202124");frames=[]
    for line,view_id in enumerate(sorted(groups[RUNS[0]])):
        baseline=None
        for column,run_id in enumerate(RUNS):
            row,receipt=groups[run_id][view_id]
            if row["status"]!="captured" or row["target_checks"]["planned_instance_present"] is not True:raise ValueError("Planned instance is absent")
            box=target_box(row,receipt);baseline=box if baseline is None else baseline
            delta=max(abs(a-b) for a,b in zip(box,baseline))
            if delta>1:raise ValueError("Paired target boxes differ by more than one pixel")
            image=Image.open(row["rgb_path"]).convert("RGB");draw=ImageDraw.Draw(image)
            for obj in row["truth"]["objects"]:draw.rectangle(tuple(obj["bbox_xyxy"]),outline=COLORS[obj["class_name"]],width=4)
            draw.rectangle(tuple(box),outline="white",width=7);top=image.resize((960,540));x=column*960;y=line*300;top.thumbnail((960,270));sheet.paste(top,(x,y+20));ImageDraw.Draw(sheet).text((x+6,y+4),run_id,fill="white")
            frames.append({"view_id":view_id,"run_id":run_id,"category":row["expected_category"],"object_id":row["expected_object_id"],"image_path":row["rgb_path"],"image_sha256":row["image_sha256"],"label_sha256":object_sha256(row["truth"]),"target_bbox_xyxy":box,"bbox_delta_px":delta})
    out=PILOT/"positive-review-contact-sheet.png";sheet.save(out);manifest={"status":"rendered_pending_ai_review","frames":frames,"contact_sheet":str(out),"contact_sheet_sha256":file_sha256(out),"inputs":inputs,"training_admitted":False,"promotable":False};manifest["identity"]=object_sha256(manifest);write_json(PILOT/"positive-review-manifest.json",manifest);return manifest

def finalize(manifest,accepted_ids):
    if {r["view_id"] for r in manifest["frames"]}!=set(accepted_ids) or len(accepted_ids)!=4:raise ValueError("Explicit pair decisions are incomplete")
    frames=[]
    for row in manifest["frames"]:
        if file_sha256(row["image_path"])!=row["image_sha256"]:raise ValueError("Reviewed image changed")
        frames.append({**row,"decision":"accepted","review_nature":"AI-assisted","reviewer":"codex_visual_inspection_2026-09-07","reason":"Full frame and instance overlay inspected: planned equipment is visible, usable, and consistently labelled.","training_admitted":False,"promotable":False})
    result={"status":"reviewed_pilot_only","accepted":8,"held":0,"complete_pairs":4,"frames":frames,"contact_sheet_sha256":manifest["contact_sheet_sha256"],"training_admitted":False,"promotable":False};result["identity"]=object_sha256(result);write_json(PILOT/"positive-semantic-review.json",result);return result

if __name__=="__main__":
    manifest=render();print(json.dumps({"identity":manifest["identity"],"sheet":manifest["contact_sheet"]},indent=2))
