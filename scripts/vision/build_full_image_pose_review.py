"""Eight-frame review evidence; retain shared-source links and never auto-admit."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.capture_local_full_image_pilot import OUT as PARENT,read,save,file_sha256,verify_tree
from src.vision.canonical.plan import read_record
from src.ml.artifacts import object_sha256
OUT=PARENT/'original-pilot-v1'

def main():
    path=OUT/'review/manifest.json'
    if path.exists():verify_tree(path);print('VERIFIED_EXISTING');return
    verify_tree(OUT/'progress.json');r=read_record(OUT/'capture/collection-receipt.json');p=read_record(OUT/'plan/plan.json')
    if len(r['views'])!=8 or any(x['status']!='captured' for x in r['views']):raise ValueError('Incomplete original pilot')
    mapping=r['collection_checks']['instance_mapping'];byid={v['view_id']:v for v in p['calibration_views']};frames=[];inputs={}
    path.parent.mkdir(parents=True,exist_ok=False)
    for n,row in enumerate(r['views'],1):
        if file_sha256(row['rgb_path'])!=row['image_sha256']:raise ValueError('Stale source image')
        inputs[row['rgb_path']]=row['image_sha256'];im=Image.open(row['rgb_path']).convert('RGB');d=ImageDraw.Draw(im);objects=[]
        for index,b in enumerate(row['raw_truth'].get('annotatedBox',[])):
            lo=b['box'].get('minCorner',{});hi=b['box'].get('maxCorner',{});box=[float(lo.get('x',0)),float(lo.get('y',0)),float(hi.get('x',0)),float(hi.get('y',0))];identity=mapping[str(b['label'])]
            planned=identity['object_id']==row['expected_object_id'];d.rectangle(box,outline='red' if planned else 'lime',width=4);d.text((max(0,box[0]),max(0,box[1])),f'{index} {identity["object_id"]}',fill='red' if planned else 'lime')
            objects.append(dict(**identity,runtime_label=b['label'],bbox_xyxy=box,planned=planned,review_status='pending'))
        op=path.parent/f'frame-{n:02}.png';im.save(op);inputs[str(op)]=file_sha256(op);view=byid[row['view_id']]
        frames.append(dict(frame_id=f'F{n:02}',view_id=row['view_id'],category=row['expected_category'],image_path=row['rgb_path'],image_sha256=row['image_sha256'],truth_sha256=object_sha256(row['raw_truth']),overlay_path=str(op),objects=objects,review_status='pending',derivation_group=view['derivation_group'],derived_from_view_id=view.get('derived_from_view_id'),shared_source_group=view.get('shared_source_group',view['family'])))
    for p in (OUT/'progress.json',OUT/'plan/plan.json',Path(__file__)):inputs[str(p)]=file_sha256(p)
    save(path,dict(status='pending_explicit_full_image_review',frames=frames,frame_count=8,box_count=sum(len(f['objects']) for f in frames),lineage_rule='Shared source group and parent-view relationship take precedence over distinct per-pose derivation_group for partitioning; do not count nearby switchgear views as independent scenes.',inputs=inputs))
    print('REVIEW_READY',8,sum(len(f['objects']) for f in frames),OUT,flush=True)

if __name__=='__main__':main()
