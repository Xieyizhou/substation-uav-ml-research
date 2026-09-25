"""Instance-bound development inference with explicit diagnostic ROI overlays."""
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record
from scripts.vision.analyze_recovery_paired_calibration import indexed,iou


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1'
    contract_path=ROOT/'config/perception/visual_experiment_baseline_v1.json'
    contract=json.loads(contract_path.read_text())
    weights=ROOT/contract['model']['package_root']/'weights/best.pt'
    assert file_sha256(weights)==contract['model']['weights_sha256']
    model=YOLO(str(weights));assert [model.names[i] for i in range(4)]==contract['class_order']
    progress_path=base/'control-matrix-capture-v1/progress.json'
    inputs={str(p):file_sha256(p) for p in (contract_path,weights,progress_path,Path(__file__))}
    frames=[]
    for run in json.loads(progress_path.read_text())['runs']:
        plan=read_record(run['plan_path']);receipt=read_record(run['receipt_path'])
        assert receipt['status']=='complete_pending_review' and receipt['plan_identity']==plan['identity']
        for p in (run['plan_path'],run['receipt_path']):inputs[p]=file_sha256(Path(p))
        views={v['view_id']:v for v in plan['calibration_views']};objects={o['name']:o for o in plan['objects']}
        for row in receipt['views']:
            path=Path(row['rgb_path']);inputs[str(path)]=file_sha256(path);assert inputs[str(path)]==row['image_sha256']
            view=views[row['view_id']];obj=objects[view['object_id']]
            labels=set(map(int,obj['runtime_labels']));truth=indexed(row['truth']['objects'])
            targets=[o for label in labels for o in truth.get(label,[])]
            with Image.open(path) as im:
                result=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,batch=1,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
            predictions=[{'bbox_xyxy':b,'confidence':c,'class_name':model.names[int(k)]} for b,c,k in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist())]
            frames.append({'index':len(frames)+1,'batch':run['name'],'view':view,'rgb_path':str(path),'target_truth':targets,'predictions':predictions,
                           'correct_instance_hit':len(targets)==1 and any(p['class_name']==view['category'] and iou(p['bbox_xyxy'],targets[0]['bbox_xyxy'])>=.5 for p in predictions)})
    assert len(frames)==80
    out=base/'matrix-baseline-v1';out.mkdir(exist_ok=True)
    visuals=ROOT/'outputs/research/ml_training_recovery_v1/matrix-baseline-v1';visuals.mkdir(parents=True,exist_ok=True)
    sheets={}
    for start in range(0,80,8):
        page=Image.new('RGB',(1280,1600),'#202020');draw=ImageDraw.Draw(page)
        for j,frame in enumerate(frames[start:start+8]):
            x,y=(j%2)*640,(j//2)*400
            with Image.open(frame['rgb_path']) as im:page.paste(im.resize((640,360)),(x,y+40))
            draw.text((x+3,y+3),f'{frame["index"]} {frame["batch"]} {frame["view"]["object_id"]}',fill='white')
            draw.text((x+3,y+20),'cyan: occupancy ROI | green: instance truth | orange: prediction',fill='white')
            boxes=[(frame['view']['diagnostic_projected_aabb'],'cyan','ROI')]+[(o['bbox_xyxy'],'lime','truth') for o in frame['target_truth']]+[(p['bbox_xyxy'],'orange',f'{p["class_name"]} {p["confidence"]:.2f}') for p in frame['predictions']]
            for b,color,label in boxes:
                a,b0,c,d=b;draw.rectangle((x+a/3,y+40+b0/3,x+c/3,y+40+d/3),outline=color,width=2)
                draw.text((x+a/3,y+40+b0/3),label,fill=color)
        path=visuals/f'page-{start//8+1:02}.jpg';page.save(path,quality=95);sheets[str(path)]=file_sha256(path)
    report={'inputs':inputs,'frames':frames,'visuals':sheets,'training_admitted':False,'status':'inference_complete_pending_visual_review',
            'settings':{'imgsz':640,'conf':.37,'iou':.7,'batch':1,'rect':False,'device':'cpu','agnostic_nms':False,'max_det':300,'matching_iou':.5},
            'ultralytics_version':__import__('ultralytics').__version__,
            'limits':['Occupancy ROI is not a ground-truth visible box.','Instance truth presence does not certify visible pixels.','Development diagnostic only; not runtime equivalence or qualification.']}
    report['identity']=object_sha256(report);write_json(out/'report.json',report)
    print('Completed',len(frames),'frames; sheets:',visuals)


if __name__=='__main__':main()
