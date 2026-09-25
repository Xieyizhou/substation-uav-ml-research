"""Render, review, deduplicate and freeze all 48 hard-negative v2 frames."""
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_hard_negative_v2 import BASE

VARIANTS=("light_normal","light_cool_low")

def load_rows():
    pilot=json.loads((BASE/"pilot-v1/capture-progress.json").read_text());remaining=json.loads((BASE/"remaining-v1/capture-progress.json").read_text())
    rows={v:{} for v in VARIANTS};inputs={}
    for progress in (pilot,remaining):
        for run in progress["runs"]:
            variant="light_cool_low" if "cool_low" in run["run_id"] else "light_normal";path=Path(run["receipt_path"]);inputs[str(path)]=file_sha256(path);receipt=json.loads(path.read_text())
            for row in receipt["views"]:
                if row["status"]!="captured" or row["truth"]["objects"] or row["target_checks"]["observed_instance_labels"]:raise ValueError("Non-empty or invalid negative frame")
                if row["view_id"] in rows[variant]:raise ValueError("Duplicate captured pose within variant")
                rows[variant][row["view_id"]]=row
    if any(len(group)!=24 for group in rows.values()) or set(rows[VARIANTS[0]])!=set(rows[VARIANTS[1]]):raise ValueError("48-frame paired capture is incomplete")
    return rows,inputs

def render():
    rows,inputs=load_rows();pilot_review=json.loads((BASE/"pilot-v1/semantic-review.json").read_text());pilot_ids={r["view_id"] for r in pilot_review["frames"]}
    pending=sorted(set(rows["light_normal"])-pilot_ids);out=BASE/"review-v1";out.mkdir(exist_ok=True);pages=[];frames=[]
    for page_index in range(5):
        ids=pending[page_index*4:(page_index+1)*4];sheet=Image.new("RGB",(960,1080),"#202124")
        for line,view_id in enumerate(ids):
            for column,variant in enumerate(VARIANTS):
                row=rows[variant][view_id];image=Image.open(row["rgb_path"]).convert("RGB");image.thumbnail((480,250));x=column*480;y=line*270;sheet.paste(image,(x,y+20));ImageDraw.Draw(sheet).text((x+5,y+4),f"{variant} {row['expected_object_id']}",fill="white")
                frames.append({"view_id":view_id,"variant":variant,"subject":row["expected_object_id"],"image_path":row["rgb_path"],"image_sha256":row["image_sha256"],"label_sha256":object_sha256(row["truth"]),"truth_object_count":0})
        page=out/f"page-{page_index+1}.png";sheet.save(page);pages.append({"path":str(page),"sha256":file_sha256(page),"view_ids":ids})
    manifest={"status":"rendered_pending_ai_review","previously_reviewed_pilot_ids":sorted(pilot_ids),"pending_frames":frames,"pages":pages,"inputs":inputs,"training_admitted":False,"promotable":False};manifest["identity"]=object_sha256(manifest);write_json(out/"manifest.json",manifest);return manifest

def pixel_hash(path):
    with Image.open(path) as image:return hashlib.sha256(image.convert("RGB").tobytes()+f"{image.width}x{image.height}".encode()).hexdigest()

def finalize(manifest,accepted_ids):
    expected={r["view_id"] for r in manifest["pending_frames"]}
    if expected!=set(accepted_ids) or len(expected)!=20:raise ValueError("Explicit pending-pair decisions are incomplete")
    rows,_=load_rows();pilot=json.loads((BASE/"pilot-v1/semantic-review.json").read_text());decisions=list(pilot["frames"]);file_seen={};pixel_seen={};duplicates=[]
    for view_id in sorted(expected):
        for variant in VARIANTS:
            row=rows[variant][view_id];path=Path(row["rgb_path"])
            if file_sha256(path)!=row["image_sha256"]:raise ValueError("Image hash changed after capture")
            ph=pixel_hash(path);exact=file_seen.get(row["image_sha256"]);same_pixels=pixel_seen.get(ph)
            if exact or same_pixels:duplicates.append({"view_id":view_id,"variant":variant,"exact_file_of":exact,"pixel_duplicate_of":same_pixels})
            file_seen[row["image_sha256"]]=f"{view_id}:{variant}";pixel_seen[ph]=f"{view_id}:{variant}"
            decisions.append({"view_id":view_id,"variant":variant,"subject":row["expected_object_id"],"image_path":row["rgb_path"],"image_sha256":row["image_sha256"],"pixel_sha256":ph,"label_sha256":object_sha256(row["truth"]),"truth_object_count":0,"decision":"accepted","review_nature":"AI-assisted","reviewer":"codex_visual_inspection_2026-09-07","reason":"Full frame inspected: non-target infrastructure is visible, image quality is usable, and full_2d truth is empty.","training_admitted":False,"promotable":False})
    if duplicates:raise ValueError(f"Unexpected exact/pixel duplicates: {duplicates}")
    review={"status":"reviewed","accepted":48,"held":0,"complete_pose_pairs":24,"frames":decisions,"unexpected_duplicates":[],"designed_lineage_pairs":24,"training_admitted":False,"promotable":False};review["identity"]=object_sha256(review);write_json(BASE/"semantic-review.json",review)
    original=json.loads((BASE/"intake-ledger.json").read_text());by={(r["pose_id"],r["lighting_id"]):r for r in original["entries"]}
    for row in decisions:
        item=by[(row["view_id"],row["variant"])];item.update(image_sha256=row["image_sha256"],label_sha256=row["label_sha256"],instance_present=False,visibility_status="not_applicable",truncation_status="not_applicable",capture_status="captured",review_decision="accepted",review_reason=row["reason"],review_nature="AI-assisted",reviewed_at="2026-09-07T00:00:00+08:00",dedup_status="designed_variant",admission_status="admitted",failure_reason=None)
    ledger={**original,"status":"frozen","entries":list(by.values()),"source_review_identity":review["identity"]};ledger.pop("identity",None);ledger["identity"]=object_sha256(ledger);write_json(BASE/"frozen-intake-ledger.json",ledger)
    return review,ledger

if __name__=="__main__":
    manifest=render();print(json.dumps({"manifest":manifest["identity"],"pages":manifest["pages"]},indent=2))
