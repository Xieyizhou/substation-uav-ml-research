"""Render the review-only hard-negative crops for semantic inspection."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


def main():
    base=ROOT/'data/research/ml_training_recovery_v1/background-hard-negative-candidates-v1'
    manifest=json.loads((base/'manifest.json').read_text())
    visual=ROOT/'outputs/research/ml_training_recovery_v1/background-hard-negative-candidates-v1.jpg'
    visual.parent.mkdir(parents=True,exist_ok=True)
    sheet=Image.new('RGB',(1200,1500),'#202020');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(manifest['rows']):
        x=(i%3)*400;y=(i//3)*290
        with Image.open(row['crop_path']) as im:
            im.thumbnail((390,245));sheet.paste(im,(x,y+35))
        draw.text((x+4,y+5),f"{row['candidate_id']} {row['prediction']['class_name']} {row['prediction']['confidence']:.3f}",fill='white')
        draw.text((x+4,y+20),'candidate crop; source frame has other labels',fill='#bbbbbb')
    sheet.save(visual,quality=95)
    output={'input_manifest':str(base/'manifest.json'),'input_manifest_sha256':file_sha256(base/'manifest.json'),
            'visual':str(visual),'visual_sha256':file_sha256(visual),'status':'ready_for_semantic_review',
            'training_admitted':False,'method':'Contact sheet only; no automatic admission.'}
    output['identity']=object_sha256(output);write_json(base/'review-sheet.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
