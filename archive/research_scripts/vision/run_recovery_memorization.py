"""Non-promotable training-set memorization diagnostic; no holdout evaluation."""
import json
import sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1';out=base/'memorization-v1'
    out.mkdir(exist_ok=False)
    contract_path=ROOT/'config/perception/visual_experiment_baseline_v1.json';contract=json.loads(contract_path.read_text())
    weights=ROOT/contract['model']['package_root']/'weights/best.pt';assert file_sha256(weights)==contract['model']['weights_sha256']
    controls=json.loads((base/'matrix-baseline-v1/report.json').read_text())
    excluded={file_sha256(Path(r['rgb_path'])) for r in controls['frames']}
    source=base/'paired-calibration-v1/plan-manifest.json';selected=[]
    for run in json.loads(source.read_text())['runs']:
        if run['mode']!='full_2d':continue
        receipt=base/'paired-calibration-v1'/run['name']/'capture/collection-receipt.json'
        for row in json.loads(receipt.read_text())['views']:
            if row['view_id'] not in run['held_target_view_ids'] or row['image_sha256'] in excluded:continue
            if not all(2<=o['bbox_xyxy'][0]<o['bbox_xyxy'][2]<=1918 and 2<=o['bbox_xyxy'][1]<o['bbox_xyxy'][3]<=1078 for o in row['truth']['objects']):continue
            assert file_sha256(Path(row['rgb_path']))==row['image_sha256']
            selected.append({'row':row,'receipt_path':str(receipt),'receipt_sha256':file_sha256(receipt),'plan_path':run['plan_path'],'plan_sha256':file_sha256(Path(run['plan_path']))})
    assert len(selected)==18
    images=out/'images';labels=out/'labels';images.mkdir();labels.mkdir()
    for n,item in enumerate(selected):
        row=item['row']
        with Image.open(row['rgb_path']) as im:im.save(images/f'{n:03}.png')
        lines=[]
        for obj in row['truth']['objects']:
            a,b,c,d=obj['bbox_xyxy'];k=contract['class_order'].index(obj['class_name'])
            lines.append(f'{k} {(a+c)/3840:.10f} {(b+d)/2160:.10f} {(c-a)/1920:.10f} {(d-b)/1080:.10f}')
        (labels/f'{n:03}.txt').write_text('\n'.join(lines)+'\n')
    dataset=out/'dataset.yaml';dataset.write_text(f'path: {out}\ntrain: images\nval: images\nnames: '+json.dumps(contract['class_order'])+'\n')
    protocol={'purpose':'training_set_memorization_only','selected':selected,'formal_training_admitted':False,
              'diagnostic_training_only':True,'promotable':False,'holdout_evaluation':False,
              'limits':['Train and val intentionally identical: memorization metrics only.',
                        'Scenes overlap prior development diagnostics; not independent evaluation.',
                        'No full new protected near-duplicate certification; artifacts cannot enter qualification.',
                        'All original object boxes preserved; no cabinet ROI labels added.'],
              'weights_sha256':file_sha256(weights),'script_sha256':file_sha256(Path(__file__))}
    protocol['identity']=object_sha256(protocol);write_json(out/'protocol.json',protocol)
    model=YOLO(str(weights))
    model.train(data=str(dataset),epochs=10,imgsz=640,batch=6,device='cpu',workers=0,
                optimizer='AdamW',lr0=.001,lrf=1.0,warmup_epochs=0,warmup_bias_lr=0,
                seed=7,deterministic=True,patience=0,amp=False,
                mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
                flipud=0,fliplr=0,hsv_h=0,hsv_s=0,hsv_v=0,
                project=str(out),name='fit',plots=False,save=True,val=False)
    write_json(out/'completion.json',{'status':'diagnostic_fit_complete','promotable':False,
                                     'formal_training_admitted':False,'epochs':10,'frames':18})


if __name__=='__main__':main()
