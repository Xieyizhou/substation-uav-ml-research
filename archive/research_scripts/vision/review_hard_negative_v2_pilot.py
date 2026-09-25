"""Render and finalize the explicitly inspected eight-frame v2 negative pilot."""
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_hard_negative_v2 import BASE

def render():
    progress=json.loads((BASE/"pilot-v1/capture-progress.json").read_text());groups=[];inputs={}
    for run in progress["runs"]:
        path=Path(run["receipt_path"]);inputs[str(path)]=file_sha256(path);receipt=json.loads(path.read_text())
        rows=[]
        for row in receipt["views"]:
            if row["status"]!="captured" or row["truth"]["objects"] or row["target_checks"]["observed_instance_labels"]:raise ValueError("Pilot frame is not a true full-image negative")
            rows.append(row)
        groups.append(rows)
    if len(groups)!=2 or any(len(rows)!=4 for rows in groups):raise ValueError("Pilot is incomplete")
    sheet=Image.new("RGB",(960,1080),"#202124");frames=[]
    for column,rows in enumerate(groups):
        for index,row in enumerate(rows):
            image=Image.open(row["rgb_path"]).convert("RGB");image.thumbnail((480,250));x=column*480;y=index*270
            sheet.paste(image,(x,y+20));ImageDraw.Draw(sheet).text((x+6,y+4),("normal" if column==0 else "cool_low")+" "+row["expected_object_id"],fill="white")
            frames.append({"view_id":row["view_id"],"variant":"light_normal" if column==0 else "light_cool_low","subject":row["expected_object_id"],"image_path":row["rgb_path"],"image_sha256":row["image_sha256"],"label_sha256":object_sha256(row["truth"]),"truth_object_count":0})
    out=BASE/"pilot-v1/review-contact-sheet.png";sheet.save(out)
    manifest={"status":"rendered_pending_ai_review","frames":frames,"contact_sheet":str(out),"contact_sheet_sha256":file_sha256(out),"inputs":inputs,"training_admitted":False,"promotable":False};manifest["identity"]=object_sha256(manifest);write_json(BASE/"pilot-v1/review-manifest.json",manifest);return manifest

def finalize(manifest,accepted_ids):
    if {r["view_id"] for r in manifest["frames"]}!=set(accepted_ids):raise ValueError("Explicit review decisions are incomplete")
    frames=[]
    for row in manifest["frames"]:
        if file_sha256(row["image_path"])!=row["image_sha256"]:raise ValueError("Reviewed image changed")
        frames.append({**row,"decision":"accepted","review_nature":"AI-assisted","reviewer":"codex_visual_inspection_2026-09-07","reason":"Full frame inspected: non-target infrastructure is visible, image quality is usable, and the hash-bound full_2d truth is empty.","training_admitted":False,"promotable":False})
    result={"status":"reviewed_pilot_only","accepted":8,"held":0,"frames":frames,"contact_sheet_sha256":manifest["contact_sheet_sha256"],"training_admitted":False,"promotable":False};result["identity"]=object_sha256(result);write_json(BASE/"pilot-v1/semantic-review.json",result);return result

if __name__=="__main__":
    manifest=render();print(json.dumps({"manifest":manifest["identity"],"sheet":manifest["contact_sheet"]},indent=2))
