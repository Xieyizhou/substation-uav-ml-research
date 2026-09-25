"""Create review-only crops for background hallucination candidates."""
import json
import sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json


def main():
    base=ROOT/'data/research/ml_training_recovery_v1/memorization-v1/unmatched-audit-v1'
    source=base/'final-review.json';review=json.loads(source.read_text())
    out=ROOT/'data/research/ml_training_recovery_v1/background-hard-negative-candidates-v1';out.mkdir(exist_ok=False)
    crops=out/'crops';crops.mkdir()
    inputs={str(source):file_sha256(source),__file__:file_sha256(Path(__file__))};rows=[]
    for item in review['queue']:
        if item['visual_category']!='confirmed_background_hallucination':continue
        path=Path(item['image_path']);inputs[str(path)]=file_sha256(path)
        with Image.open(path) as im:
            a,b,c,d=item['prediction']['bbox_xyxy'];w,h=im.size
            bw,bh=c-a,d-b;left=max(0,int(a-.5*bw));top=max(0,int(b-.5*bh));right=min(w,int(c+.5*bw));bottom=min(h,int(d+.5*bh))
            crop=im.crop((left,top,right,bottom));crop_path=crops/f'{item["review_index"]:02}.png';crop.save(crop_path)
        rows.append({'candidate_id':f'unmatched-{item["review_index"]:02}','source_image':str(path),
                     'source_image_sha256':inputs[str(path)],'prediction':item['prediction'],
                     'crop_path':str(crop_path),'crop_sha256':file_sha256(crop_path),
                     'source_truth_objects':item['truth'],'visual_status':'background_hallucination_confirmed',
                     'training_admitted':False,'requires_review_before_training':True,
                     'reason':'Region is negative evidence only; the source frame contains other labeled equipment and cannot be treated as a whole-frame negative.'})
    manifest={'schema_version':1,'status':'review_only_hard_negative_candidates','rows':rows,'count':len(rows),
              'inputs':inputs,'training_admitted':False,
              'limits':['Crops are derived from model predictions and assistant visual review.',
                        'No crop is admitted to training by this manifest.',
                        'Source frames contain positive equipment labels; only the crop region is a candidate negative.',
                        'A human semantic review is still required before use.']}
    manifest['identity']=object_sha256(manifest);write_json(out/'manifest.json',manifest)
    print(json.dumps({'count':len(rows),'output':str(out/'manifest.json'),'identity':manifest['identity']}))


if __name__=='__main__':main()
