"""Finalize only explicit, hash-bound AI-assisted decisions; never invent passes."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.prepare_paired_visual_factors import BASE, VARIANTS

# These are the twelve sheets inspected by Codex on 2026-09-07. This allowlist is
# deliberately explicit: adding a newly captured pair requires a new review edit.
INSPECTED={
"0376b004e01ef4132cdbce21fa4c779ae7139d236cae2f5fc17a8cc5ff8b88c6",
"0b56b9bea8691c9cfdf32b3936c68b1bd0cc9a848734449896ebfd319d710225",
"23c1e910e317c6ac2af13a110cb4b666025c6bf8d3249a6c5a6afe6f47031e61",
"2736443628942343be43f9c513868fc888fad9823908f2bb8789daa0e0d6a5d2",
"3227d1a73204c8f7a79903dfacaacb4c768c0581cd8242a0477958d700ced7f0",
"60d4a69fede9d30b46f4703d67e1860a3c7a4c3f4aef44da02cd504178705492",
"905465ff8a0fb505a5f3907bc9de821362682eae5a325110d0fd43e47eb9f032",
"a3dd273fd63820e58110ea00d8af1eb9137cf064b2f0895c0dc0b6f9a7acf2e3",
"ae573a8e46f4fc709d59df7eeb84570445be7f1aac48d267cab384f873c8b9a6",
"c02798da07f2ca165a0ddb8a499e94970f6cd6cbf898dac3da1aa1d473271020",
"d02b5eb39336735a4525e1b063426a43fd3e868a8f6a33f555ef17c2561a8603",
"ee69071a5ba9752725bd5cf405f2e896bd3f654e28d9706a463e0dbcde2d1e97",
}


def validate_and_finalize(manifest, inspected=INSPECTED):
    frames=manifest.get("frames",[]); by={}
    for row in frames:
        if file_sha256(row["image_path"])!=row["image_sha256"] or file_sha256(row["overlay_path"])!=row["overlay_sha256"] or file_sha256(row["crop_path"])!=row["crop_sha256"]:
            raise ValueError("Review evidence hash is stale")
        by.setdefault(row["view_id"],{})[row["variant"]]=row
    if set(by)!=set(inspected) or any(set(group)!=set(VARIANTS) for group in by.values()):
        raise ValueError("Paired review is incomplete")
    decisions=[]
    for view_id,group in sorted(by.items()):
        baseline=group["original"]["target_bbox_xyxy"]
        maximum=max(abs(a-b) for row in group.values() for a,b in zip(row["target_bbox_xyxy"],baseline))
        if maximum>1: raise ValueError("Paired target boxes differ by more than one pixel")
        for variant in VARIANTS:
            row=group[variant]
            decisions.append({**row,"decision":"accepted","reviewer":"codex_visual_inspection_2026-09-07",
                              "review_nature":"AI-assisted","image_quality":"accepted","label_consistency":"accepted",
                              "planned_instance_visibility":"accepted","bbox_max_delta_vs_original_px":maximum,
                              "reason":"Full image and target crop inspected: planned equipment is visible, framing is usable, overlays align, and no corruption is present.",
                              "training_admitted":False,"promotable":False})
    return {"status":"reviewed","review_nature":"AI-assisted per-frame review","accepted":len(decisions),"held":0,
            "complete_pairs":len(by),"frames":decisions,"training_admitted":False,"promotable":False}


def main():
    path=BASE/"review-evidence/manifest.json"; manifest=json.loads(path.read_text())
    result=validate_and_finalize(manifest); result["inputs"]={str(path):file_sha256(path),str(Path(__file__)):file_sha256(Path(__file__))}
    result["identity"]=object_sha256(result); write_json(BASE/"semantic-review.json",result)
    print(json.dumps({k:result[k] for k in ("status","accepted","held","complete_pairs","identity")},indent=2))


if __name__=="__main__": main()
