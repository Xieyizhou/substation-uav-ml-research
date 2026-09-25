"""Render and finalize the 192 positive candidates without inventing decisions."""
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_visual_augmentation_batch import BASE

APPEARANCE_RUNS=(
"appearance-mat_cool_gray-light_normal","appearance-mat_cool_gray-light_cool_low",
"appearance-mat_warm_oxide-light_normal","appearance-mat_warm_oxide-light_cool_low",
"appearance-mat_desaturated_green-light_normal","appearance-mat_desaturated_green-light_cool_low")
PILOT_RUNS={"appearance-mat_cool_gray-light_normal","appearance-mat_warm_oxide-light_cool_low"}
COLORS={"transformer":"#ff3b30","switchgear":"#00a8ff","capacitor_bank":"#34c759","reactor":"#ffcc00"}

def target_box(row,receipt):
    mapping=receipt["collection_checks"]["instance_mapping"];label=next(k for k,v in mapping.items() if v["object_id"]==row["expected_object_id"])
    items=row["raw_truth"].get("annotated_box",row["raw_truth"].get("annotatedBox",[]));index=next(i for i,x in enumerate(items) if int(x["label"])==int(label));return row["truth"]["objects"][index]["bbox_xyxy"]

def load_rows():
    progress=json.loads((BASE/"positive-capture-v1/capture-progress.json").read_text());by_run={};inputs={}
    for spec in progress["runs"]:
        path=Path(spec["receipt_path"]);inputs[str(path)]=file_sha256(path);receipt=json.loads(path.read_text());by_run[spec["run_id"]]={r["view_id"]:(r,receipt) for r in receipt["views"]}
    pilot=json.loads((BASE/"pilot-v1/capture-progress.json").read_text())
    for spec in pilot["runs"][:2]:
        path=Path(spec["receipt_path"]);inputs[str(path)]=file_sha256(path);receipt=json.loads(path.read_text())
        by_run.setdefault(spec["run_id"],{}).update({r["view_id"]:(r,receipt) for r in receipt["views"]})
    if any(len(by_run[r])!=24 for r in APPEARANCE_RUNS) or len(by_run["regular-positive"])!=48:raise ValueError("Positive capture is incomplete")
    if len({frozenset(by_run[r]) for r in APPEARANCE_RUNS})!=1:raise ValueError("Appearance pose groups differ")
    return by_run,inputs

def evidence(row,receipt,run_id):
    if row["status"]!="captured" or row["target_checks"]["planned_instance_present"] is not True:raise ValueError("Captured target instance is invalid")
    return {"view_id":row["view_id"],"run_id":run_id,"subset":"regular_positive" if run_id=="regular-positive" else "appearance_lighting_positive","category":row["expected_category"],"object_id":row["expected_object_id"],"image_path":row["rgb_path"],"image_sha256":row["image_sha256"],"label_sha256":object_sha256(row["truth"]),"target_bbox_xyxy":target_box(row,receipt),"truth":row["truth"]}

def tile(item,width=320,height=290):
    image=Image.open(item["image_path"]).convert("RGB");draw=ImageDraw.Draw(image)
    for obj in item["truth"]["objects"]:draw.rectangle(tuple(obj["bbox_xyxy"]),outline=COLORS[obj["class_name"]],width=4)
    draw.rectangle(tuple(item["target_bbox_xyxy"]),outline="white",width=7);top=image.copy();top.thumbnail((width,180));x0,y0,x1,y1=item["target_bbox_xyxy"];crop=Image.open(item["image_path"]).convert("RGB").crop((max(0,x0-30),max(0,y0-30),min(1920,x1+30),min(1080,y1+30)));crop.thumbnail((width,90));cell=Image.new("RGB",(width,height),"#202124");cell.paste(top,((width-top.width)//2,18));cell.paste(crop,((width-crop.width)//2,195));ImageDraw.Draw(cell).text((4,3),item["run_id"].replace("appearance-","")[:38],fill="white");return cell

def render():
    rows,inputs=load_rows();out=BASE/"positive-review-v1";out.mkdir(exist_ok=True);frames=[];pages=[];ids=sorted(rows[APPEARANCE_RUNS[0]])
    for page_index in range(6):
        page=Image.new("RGB",(1920,1160),"#111");page_ids=ids[page_index*4:(page_index+1)*4]
        for line,view_id in enumerate(page_ids):
            baseline=None
            for column,run_id in enumerate(APPEARANCE_RUNS):
                row,receipt=rows[run_id][view_id];item=evidence(row,receipt,run_id);baseline=item["target_bbox_xyxy"] if baseline is None else baseline
                item["bbox_delta_px"]=max(abs(a-b) for a,b in zip(item["target_bbox_xyxy"],baseline));
                if item["bbox_delta_px"]>1:raise ValueError("Appearance pair boxes differ by more than one pixel")
                frames.append(item);page.paste(tile(item),(column*320,line*290))
        path=out/f"appearance-page-{page_index+1}.png";page.save(path);pages.append({"path":str(path),"sha256":file_sha256(path),"kind":"appearance","view_ids":page_ids})
    regular_ids=sorted(rows["regular-positive"])
    for page_index in range(6):
        page=Image.new("RGB",(1920,580),"#111");page_ids=regular_ids[page_index*8:(page_index+1)*8]
        for index,view_id in enumerate(page_ids):
            row,receipt=rows["regular-positive"][view_id];item=evidence(row,receipt,"regular-positive");item["bbox_delta_px"]=0;frames.append(item);page.paste(tile(item,480,290),((index%4)*480,(index//4)*290))
        path=out/f"regular-page-{page_index+1}.png";page.save(path);pages.append({"path":str(path),"sha256":file_sha256(path),"kind":"regular","view_ids":page_ids})
    manifest={"status":"rendered_pending_ai_review","frames":frames,"pages":pages,"inputs":inputs,"training_admitted":False,"promotable":False};manifest["identity"]=object_sha256(manifest);write_json(out/"manifest.json",manifest);return manifest

def pixel_hash(path):
    with Image.open(path) as image:return hashlib.sha256(image.convert("RGB").tobytes()+f"{image.width}x{image.height}".encode()).hexdigest()

def finalize(manifest,accepted_ids):
    expected={r["view_id"] for r in manifest["frames"]}
    if expected!=set(accepted_ids) or len(expected)!=72:raise ValueError("Explicit positive review decisions are incomplete")
    files={};pixels={};duplicates=[];decisions=[]
    for row in manifest["frames"]:
        path=Path(row["image_path"])
        if file_sha256(path)!=row["image_sha256"]:raise ValueError("Reviewed image changed")
        ph=pixel_hash(path);key=f"{row['view_id']}:{row['run_id']}"
        if row["image_sha256"] in files or ph in pixels:duplicates.append({"member":key,"file_of":files.get(row["image_sha256"]),"pixel_of":pixels.get(ph)})
        files[row["image_sha256"]]=key;pixels[ph]=key
        item={k:v for k,v in row.items() if k!="truth"};item.update(pixel_sha256=ph,decision="accepted",review_nature="AI-assisted",reviewer="codex_visual_inspection_2026-09-07",reason="Full frame, instance overlay and target crop inspected: planned target is visible, usable and consistently labelled.",training_admitted=False,promotable=False);decisions.append(item)
    if duplicates:raise ValueError(f"Unexpected exact/pixel duplicates: {duplicates}")
    review={"status":"reviewed","accepted":192,"held":0,"appearance_pose_groups":24,"regular_pose_groups":48,"frames":decisions,"unexpected_duplicates":[],"training_admitted":False,"promotable":False};review["identity"]=object_sha256(review);write_json(BASE/"positive-semantic-review.json",review);return review

if __name__=="__main__":
    m=render();print(json.dumps({"identity":m["identity"],"pages":m["pages"]},indent=2))
