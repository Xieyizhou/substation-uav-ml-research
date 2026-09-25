"""Record explicit hash-bound AI-assisted decisions after visual inspection."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE

DECISIONS={
    "8805d33de7dac8de185d319b91a3dda050fc00e0da9b29b1382d5756e30e0fa9":"Planned capacitor bank is fully visible at useful far scale; image is sharp and the target box follows the complete body.",
    "25b4df7db036f1b46a1f35e28ad5e9f5b9618f0c88a7ce9162cde34f2c3d12b3":"Planned reactor is fully visible at useful far scale; image is sharp and the target box follows the cylindrical body.",
    "efd13046a1b61010915c9067f69f99c0d6a5213824cad228508887072abe4e33":"Neutral-gray capacitor bank remains clearly distinguishable, fully visible and consistently boxed.",
    "96e72e934cda3dcc44fbdf5b5a23952551d7946de50925e0d805bcfc85b5508b":"Neutral-gray reactor remains clearly distinguishable, fully visible and consistently boxed.",
    "46a26f97e78d18b968fffd8c0d1dd93d598a0c443e8a924d8ebabe3a00291271":"Background-shifted capacitor bank is fully visible with usable contrast and consistent target geometry.",
    "bf8b56679d19c2a305c8e7524f0a63b2cc1c1e0f2283ddbd25d97811f4eb5161":"Background-shifted reactor is fully visible with usable contrast and consistent target geometry.",
}


def main():
    manifest_path=BASE/"pilot-v1/review-v1/manifest.json";manifest=json.loads(manifest_path.read_text())
    if manifest["status"]!="pending_explicit_review" or {row["view_id"] for row in manifest["frames"]}!=set(DECISIONS):
        raise ValueError("Review manifest membership differs from explicit decisions")
    frames=[]
    for row in manifest["frames"]:
        for key in ("image","overlay","crop"):
            if file_sha256(Path(row[f"{key}_path"]))!=row[f"{key}_sha256"]: raise ValueError("Review evidence changed")
        frames.append({**row,"decision":"accepted","reason":DECISIONS[row["view_id"]],"image_quality":"usable_reviewed",
            "visibility_status":"visible_reviewed","truncation_status":"not_truncated_reviewed","reviewer":"codex_visual_inspection_2026-09-07",
            "reviewed_at":"2026-09-07T11:20:00+08:00"})
    by_category={}
    for category in ("capacitor_bank","reactor"):
        rows=[row for row in frames if row["expected_category"]==category];reference=rows[0]["target_bbox_xyxy"]
        by_category[category]={"variants":sorted(row["variant"] for row in rows),
            "maximum_target_bbox_delta_px":max(abs(a-b) for row in rows for a,b in zip(row["target_bbox_xyxy"],reference))}
    result={"schema_version":1,"status":"reviewed","review_nature":"AI-assisted","accepted":6,"held":0,"frames":frames,
        "paired_geometry":by_category,"inputs":{str(manifest_path):file_sha256(manifest_path),str(Path(__file__)):file_sha256(Path(__file__))},
        "unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);write_json(BASE/"pilot-v1/review-v1/semantic-review.json",result)
    print(json.dumps({"status":result["status"],"accepted":6,"held":0,"paired_geometry":by_category,"identity":result["identity"]},indent=2))


if __name__=="__main__": main()
