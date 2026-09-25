"""Build seven review sheets for the 42 remaining bridge-positive frames."""
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE


def evidence(receipt,view):
    mapping=receipt["collection_checks"]["instance_mapping"]
    labels=[int(label) for label,row in mapping.items() if row["object_id"]==view["expected_object_id"]]
    if len(labels)!=1: raise ValueError("Ambiguous target label")
    raw=next(row for row in view["raw_truth"]["annotatedBox"] if row["label"]==labels[0])["box"]
    lo,hi=raw.get("minCorner",{}),raw.get("maxCorner",{});return [float(lo.get("x",0)),float(lo.get("y",0)),float(hi.get("x",0)),float(hi.get("y",0))]


def main():
    progress_path=BASE/"remaining-positive-v1/capture-progress.json";progress=json.loads(progress_path.read_text())
    if progress["status"]!="complete_pending_review" or progress["captured"]!=42: raise ValueError("Remaining capture incomplete")
    out=BASE/"remaining-positive-v1/review-v1";out.mkdir(parents=True,exist_ok=False);frames=[];inputs={str(progress_path):file_sha256(progress_path)};tiles=[]
    for run in progress["runs"]:
        receipt_path=Path(run["receipt_path"]);receipt=json.loads(receipt_path.read_text());inputs[str(receipt_path)]=file_sha256(receipt_path)
        plan_path=receipt_path.parent.parent/"plan/plan.json";plan=json.loads(plan_path.read_text());inputs[str(plan_path)]=file_sha256(plan_path);planned={row["view_id"]:row for row in plan["calibration_views"]}
        for view in receipt["views"]:
            path=Path(view["rgb_path"]);bbox=evidence(receipt,view);image=Image.open(path).convert("RGB");draw=ImageDraw.Draw(image)
            for obj in view["truth"]["objects"]: draw.rectangle(obj["bbox_xyxy"],outline=(40,220,80),width=4)
            draw.rectangle(bbox,outline=(255,35,35),width=10);draw.text((max(0,bbox[0]),max(0,bbox[1]-20)),f"TARGET {view['expected_category']}",fill=(255,35,35))
            stem=f"{run['variant']}-{view['view_id'][:12]}";overlay=out/f"{stem}-overlay.png";crop=out/f"{stem}-target.png";image.save(overlay)
            margin=28;x1=max(0,int(bbox[0])-margin);y1=max(0,int(bbox[1])-margin);x2=min(image.width,int(bbox[2])+margin);y2=min(image.height,int(bbox[3])+margin)
            target=image.crop((x1,y1,x2,y2));target.save(crop)
            left=image.resize((640,360));right=Image.new("RGB",(320,360),(245,245,245));target.thumbnail((300,320));right.paste(target,((320-target.width)//2,30+(320-target.height)//2))
            tile=Image.new("RGB",(960,390),(255,255,255));tile.paste(left,(0,30));tile.paste(right,(640,30));ImageDraw.Draw(tile).text((8,8),f"{run['variant']} | {view['expected_category']} | {view['view_id'][:12]}",fill=(0,0,0));tiles.append(tile)
            plan_row=planned[view["view_id"]]
            frames.append({"variant":run["variant"],"view_id":view["view_id"],"pair_id":plan_row["pair_id"],"derivation_group":plan_row["derivation_group"],
                "expected_category":view["expected_category"],"expected_object_id":view["expected_object_id"],"image_path":str(path),"image_sha256":file_sha256(path),
                "truth_sha256":object_sha256(view["truth"]),"target_bbox_xyxy":bbox,"target_area_fraction":(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(image.width*image.height),
                "overlay_path":str(overlay),"overlay_sha256":file_sha256(overlay),"crop_path":str(crop),"crop_sha256":file_sha256(crop),
                "decision":"pending","reason":None,"review_nature":"AI-assisted","training_admitted":False,"promotable":False})
    sheets=[]
    for index in range(0,len(tiles),6):
        sheet=Image.new("RGB",(1920,1170),(230,230,230))
        for offset,tile in enumerate(tiles[index:index+6]): sheet.paste(tile,((offset%2)*960,(offset//2)*390))
        path=out/f"review-sheet-{index//6+1:02}.png";sheet.save(path);sheets.append({"path":str(path),"sha256":file_sha256(path),"frame_indexes":[index+1,min(index+6,len(tiles))]})
    result={"schema_version":1,"status":"pending_explicit_review","frames":frames,"sheets":sheets,"inputs":inputs,
        "unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False};result["identity"]=object_sha256(result)
    write_json(out/"manifest.json",result);print(json.dumps({"status":result["status"],"frames":len(frames),"sheets":sheets,"identity":result["identity"]},indent=2))


if __name__=="__main__": main()
