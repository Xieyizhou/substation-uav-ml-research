"""Render immutable review evidence for the paired visual-factor capture."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.prepare_paired_visual_factors import BASE, VARIANTS

COLORS={"transformer":"#ff3b30","switchgear":"#00a8ff","capacitor_bank":"#34c759","reactor":"#ffcc00"}


def receipts():
    progress=json.loads((BASE/"capture-progress.json").read_text())
    return {row["variant"]:json.loads(Path(row["receipt_path"]).read_text()) for row in progress["runs"]}


def target_box(row, receipt):
    mapping=receipt["collection_checks"]["instance_mapping"]
    label=next(key for key,value in mapping.items() if value["object_id"]==row["expected_object_id"])
    items=row["raw_truth"].get("annotated_box",row["raw_truth"].get("annotatedBox",[]))
    index=next(i for i,item in enumerate(items) if int(item["label"])==int(label))
    return row["truth"]["objects"][index]["bbox_xyxy"]


def main():
    out=BASE/"review-evidence"; out.mkdir(exist_ok=True)
    protocol=json.loads((BASE/"protocol.json").read_text())
    pair_ids={row["source_view_id"]:row["pair_id"] for row in protocol["poses"]}
    by_variant=receipts(); rows={v:{r["view_id"]:r for r in rec["views"]} for v,rec in by_variant.items()}
    evidence=[]
    for view_id in sorted(rows["original"]):
        tiles=[]; boxes={}
        for variant in VARIANTS:
            row=rows[variant][view_id]; image=Image.open(row["rgb_path"]).convert("RGB")
            draw=ImageDraw.Draw(image)
            for obj in row["truth"]["objects"]:
                box=tuple(obj["bbox_xyxy"]); draw.rectangle(box,outline=COLORS[obj["class_name"]],width=4)
            box=target_box(row,by_variant[variant]); boxes[variant]=box
            draw.rectangle(tuple(box),outline="white",width=7)
            overlay=out/f"{view_id}-{variant}-overlay.png"; image.save(overlay)
            margin=40; x0,y0,x1,y1=box
            crop=Image.open(row["rgb_path"]).convert("RGB").crop((max(0,x0-margin),max(0,y0-margin),min(1920,x1+margin),min(1080,y1+margin)))
            crop_path=out/f"{view_id}-{variant}-target.png"; crop.save(crop_path)
            top=image.resize((480,270)); bottom=crop.copy(); bottom.thumbnail((480,270))
            cell=Image.new("RGB",(480,570),"#202124"); cell.paste(top,(0,30)); cell.paste(bottom,((480-bottom.width)//2,300+(270-bottom.height)//2))
            ImageDraw.Draw(cell).text((8,8),variant,fill="white"); tiles.append(cell)
            evidence.append({"view_id":view_id,"pair_id":pair_ids[view_id],"variant":variant,
                             "expected_object_id":row["expected_object_id"],"expected_category":row["expected_category"],
                             "image_path":row["rgb_path"],"image_sha256":row["image_sha256"],
                             "truth_sha256":object_sha256(row["truth"]),"target_bbox_xyxy":box,
                             "overlay_path":str(overlay),"overlay_sha256":file_sha256(overlay),
                             "crop_path":str(crop_path),"crop_sha256":file_sha256(crop_path)})
        sheet=Image.new("RGB",(1920,570));
        for i,tile in enumerate(tiles): sheet.paste(tile,(i*480,0))
        sheet.save(out/f"{view_id}-paired.png")
    manifest={"status":"rendered_pending_ai_review","review_nature":"AI-assisted per-frame review",
              "frames":evidence,"training_admitted":False,"promotable":False}
    manifest["identity"]=object_sha256(manifest); write_json(out/"manifest.json",manifest)
    print(json.dumps({"frames":len(evidence),"pairs":len(evidence)//4,"identity":manifest["identity"]},indent=2))


if __name__=="__main__": main()
