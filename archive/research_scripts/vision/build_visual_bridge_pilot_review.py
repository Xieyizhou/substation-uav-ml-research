"""Build hash-bound overlays and target crops for the bridge pilot review."""
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE


def main():
    progress_path=BASE/"pilot-v1/capture-retry-1-progress.json";progress=json.loads(progress_path.read_text())
    if progress["status"]!="complete_pending_review": raise ValueError("Pilot capture is incomplete")
    out=BASE/"pilot-v1/review-v1";out.mkdir(parents=True,exist_ok=False);frames=[];inputs={str(progress_path):file_sha256(progress_path)}
    for run in progress["runs"]:
        receipt_path=Path(run["receipt_path"]);receipt=json.loads(receipt_path.read_text());inputs[str(receipt_path)]=file_sha256(receipt_path)
        mapping=receipt["collection_checks"]["instance_mapping"]
        for view in receipt["views"]:
            image_path=Path(view["rgb_path"]);inputs[str(image_path)]=file_sha256(image_path)
            labels=[int(label) for label,row in mapping.items() if row["object_id"]==view["expected_object_id"]]
            if len(labels)!=1: raise ValueError("Ambiguous planned-instance label")
            raw=next(row for row in view["raw_truth"]["annotatedBox"] if row["label"]==labels[0])["box"]
            lo,hi=raw.get("minCorner",{}),raw.get("maxCorner",{});target=[float(lo.get("x",0)),float(lo.get("y",0)),float(hi.get("x",0)),float(hi.get("y",0))]
            image=Image.open(image_path).convert("RGB");draw=ImageDraw.Draw(image)
            for obj in view["truth"]["objects"]: draw.rectangle(obj["bbox_xyxy"],outline=(50,220,80),width=3)
            draw.rectangle(target,outline=(255,40,40),width=8)
            draw.text((max(0,target[0]),max(0,target[1]-18)),f"TARGET {view['expected_category']} {view['expected_object_id']}",fill=(255,40,40))
            stem=f"{run['variant']}-{view['expected_category']}";overlay=out/f"{stem}-overlay.png";crop=out/f"{stem}-target.png"
            image.save(overlay);margin=24;x1=max(0,int(target[0])-margin);y1=max(0,int(target[1])-margin);x2=min(image.width,int(target[2])+margin);y2=min(image.height,int(target[3])+margin)
            image.crop((x1,y1,x2,y2)).save(crop)
            frames.append({"variant":run["variant"],"view_id":view["view_id"],"expected_category":view["expected_category"],
                "expected_object_id":view["expected_object_id"],"image_path":str(image_path),"image_sha256":file_sha256(image_path),
                "truth_sha256":object_sha256(view["truth"]),"target_bbox_xyxy":target,"target_area_fraction":(target[2]-target[0])*(target[3]-target[1])/(image.width*image.height),
                "overlay_path":str(overlay),"overlay_sha256":file_sha256(overlay),"crop_path":str(crop),"crop_sha256":file_sha256(crop),
                "decision":"pending","reason":None,"review_nature":"AI-assisted","training_admitted":False,"promotable":False})
    result={"schema_version":1,"status":"pending_explicit_review","frames":frames,"inputs":inputs,"unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False};result["identity"]=object_sha256(result);write_json(out/"manifest.json",result)
    print(json.dumps({"status":result["status"],"frames":len(frames),"identity":result["identity"],"review_dir":str(out)},indent=2))


if __name__=="__main__": main()
