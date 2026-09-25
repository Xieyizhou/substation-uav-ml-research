"""Diagnostic A/B: add semantically accepted background crops to the memorization set."""
import json
import shutil
import sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1';mem=base/'memorization-v1';neg=base/'background-hard-negative-candidates-v1'
    protocol=json.loads((mem/'protocol.json').read_text());semantic=json.loads((neg/'semantic-review.json').read_text())
    accepted=[r for r in semantic['rows'] if r['semantic_review_status']=='accepted_non_target_background_crop']
    out=base/'negative-ab-v1';out.mkdir(exist_ok=False);images=out/'images';labels=out/'labels';images.mkdir();labels.mkdir()
    names=['transformer','switchgear','capacitor_bank','reactor']
    selected=[];inputs={str(mem/'protocol.json'):file_sha256(mem/'protocol.json'),str(neg/'semantic-review.json'):file_sha256(neg/'semantic-review.json'),__file__:file_sha256(Path(__file__))}
    for i,item in enumerate(protocol['selected']):
        row=item['row'];src=Path(row['rgb_path']);dst=images/f'pos-{i:03}.png';Image.open(src).save(dst);inputs[str(src)]=file_sha256(src)
        lines=[]
        for o in row['truth']['objects']:
            a,b,c,d=o['bbox_xyxy'];k=names.index(o['class_name']);lines.append(f'{k} {(a+c)/3840:.10f} {(b+d)/2160:.10f} {(c-a)/1920:.10f} {(d-b)/1080:.10f}')
        (labels/f'pos-{i:03}.txt').write_text('\n'.join(lines)+'\n');selected.append({'kind':'positive','path':str(dst),'source_view_id':row['view_id']})
    for i,item in enumerate(accepted):
        src=Path(item['crop_path']);dst=images/f'neg-{i:03}.png';shutil.copyfile(src,dst);inputs[str(src)]=file_sha256(src);(labels/f'neg-{i:03}.txt').write_text('');selected.append({'kind':'accepted_background_crop','path':str(dst),'candidate_id':item['candidate_id']})
    dataset=out/'dataset.yaml';dataset.write_text(f'path: {out}\ntrain: images\nval: images\nnames: '+json.dumps(names)+'\n')
    protocol_ab={'selected':selected,'positive_count':18,'negative_count':len(accepted),'source_protocol':str(mem/'protocol.json'),'source_semantic_review':str(neg/'semantic-review.json'),'training_set_only':True,'promotable':False,'training_admitted':False,'inputs':inputs}
    protocol_ab['identity']=object_sha256(protocol_ab);write_json(out/'protocol.json',protocol_ab)
    weights=ROOT/'models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt';assert file_sha256(weights)=='820f882a5d375be0a294555c7168d49b6d68c21adce97f00b45fe7b0ebc54836'
    model=YOLO(str(weights));model.train(data=str(dataset),epochs=10,imgsz=640,batch=6,nbs=6,device='cpu',workers=0,optimizer='AdamW',lr0=.001,lrf=1,warmup_epochs=0,warmup_bias_lr=0,seed=7,deterministic=True,patience=0,amp=False,mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,hsv_h=0,hsv_s=0,hsv_v=0,project=str(out),name='fit',plots=False,save=True,val=False)
    write_json(out/'completion.json',{'status':'complete','weights':str(out/'fit/weights/last.pt'),'weights_sha256':file_sha256(out/'fit/weights/last.pt'),'positive_count':18,'negative_count':len(accepted),'promotable':False,'training_admitted':False})
    print(json.dumps({'positive_count':18,'negative_count':len(accepted),'output':str(out/'fit')}))


if __name__=='__main__':main()
