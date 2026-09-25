"""Label-derived size census; never infer occlusion or visible side from boxes."""
import sys
from pathlib import Path
from collections import Counter
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_stratified_negative_control import OUT,KEY,read,save,file_sha256,verify_tree

def main():
    pp=OUT/'protocol.json';verify_tree(pp);p=read(pp);counts=Counter(p['schedules'][KEY]);rows=[];bins=Counter();weighted=Counter();inputs={str(pp):file_sha256(pp),str(Path(__file__)):file_sha256(Path(__file__))}
    for member in p['pool_rows']:
        exposure=counts[member['member_id']]
        if not exposure or not member['class_instances'].get('switchgear'):continue
        for kind in ('image','label'):
            path=member[kind+'_path'];digest=file_sha256(path)
            if digest!=member[kind+'_sha256']:raise ValueError('Stale member')
            inputs[path]=digest
        with Image.open(member['image_path']) as im:w,h=im.size
        for idx,line in enumerate(Path(member['label_path']).read_text().splitlines()):
            cls,x,y,bw,bh=map(float,line.split())
            if int(cls)!=1:continue
            scale=640/max(w,h);width=bw*w*scale;height=bh*h*scale
            bucket='short_side_lt32' if min(width,height)<32 else 'short_side_32_to64' if min(width,height)<64 else 'short_side_ge64'
            bins[bucket]+=1;weighted[bucket]+=exposure
            rows.append(dict(member_id=member['member_id'],subset=member['subset'],lineage_id=member['lineage_id'],image_path=member['image_path'],image_sha256=member['image_sha256'],label_path=member['label_path'],label_sha256=member['label_sha256'],label_line_index=idx,
                bbox_yolo=[x,y,bw,bh],model_640_box_size=[width,height],size_bin=bucket,image_exposures=exposure,
                occlusion='unknown',front_panel_visibility='unknown',view_side='unknown',additional_condition_review='pending_not_training_admission'))
    if sum(weighted.values())!=p['exposures'][KEY]['class_instance_exposure']['switchgear']:raise ValueError('Class supervision census mismatch')
    save(OUT/'switchgear-condition-audit.json',dict(status='size_census_complete_visual_condition_review_pending',rows=rows,
        unique_frames=len({r['member_id'] for r in rows}),label_instances=len(rows),size_counts=dict(bins),exposure_weighted_size_counts=dict(weighted),
        scope='Existing approved training data unchanged. New occlusion/side/panel audit is pending; boxes do not establish these conditions. Bin counts are descriptive, not independent scene counts.',inputs=inputs))
    print('CENSUS',len(rows),bins,weighted)

if __name__=='__main__':main()
