"""Merge existing decisions plus four individually inspected replacement frames."""
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.hard_negative_coverage import OUT,read,save,verify,file_sha256
from scripts.vision.hard_negative_coverage_admission import audit,admit

def main():
    repair_path=OUT/'repair-review-manifest.json';repair=read(repair_path);verify(repair)
    if len(repair['frames'])!=4:raise ValueError('Unexpected repair frames')
    decisions=[]
    for r in repair['frames']:
        if r['ordinal']==34:
            kind,box,reason='mixed_structure',[274,14,815,720],'近景普通柜体遮挡杆体下部，杆体遮挡后方建筑；三种实际结构清楚，未见四类目标'
        elif r['ordinal']==41:
            kind,box,reason='cabinet',[953,293,1280,614],'右缘普通柜体明显截断，背面和底座可辨；场内另有杆体、网格和边界墙，未见四类目标'
        else:raise ValueError('Unknown replacement')
        decisions.append(dict(view_id=r['view_id'],image_sha256=r['image_sha256'],decision='accepted',no_target_visible=True,
            coverage_confirmed=True,review_nature='AI-assisted',reviewer='Codex individual visual inspection',
            reviewed_at=datetime.now(timezone.utc).isoformat(),reason=f'{r["variant"]} 已分别查看：{reason}',
            rois=[dict(content=kind,bbox_xyxy=[round(v*1.5) for v in box],visibility='visible',truncation='right' if r['ordinal']==41 else 'foreground_bottom')]))
    save(OUT/'repair-decisions.json',dict(status='explicitly_reviewed',decisions=decisions,
        inputs={str(repair_path):file_sha256(repair_path),str(Path(__file__)):file_sha256(Path(__file__))}))
    frames=list(repair['frames']);all_decisions=list(decisions);inputs={}
    excluded=read(OUT/'repair-v2.json')['rejected_view_ids']
    for stage in ('pilot','remaining','repair'):
        for suffix in ('review-manifest','decisions'):
            p=OUT/f'{stage}-{suffix}.json';doc=read(p);verify(doc);inputs[str(p)]=file_sha256(p)
        if stage=='repair':continue
        frames.extend(r for r in read(OUT/f'{stage}-review-manifest.json')['frames'] if r['view_id'] not in excluded)
        all_decisions.extend(r for r in read(OUT/f'{stage}-decisions.json')['decisions'] if r['view_id'] not in excluded)
    if len(frames)!=96 or len(all_decisions)!=96:raise ValueError('Final membership incomplete')
    save(OUT/'final-review-manifest.json',dict(status='explicit_reviews_available',frames=frames,inputs=inputs,
        excluded_view_ids=excluded,replacement_protocol_identity=read(OUT/'repair-v2.json')['identity']))
    save(OUT/'final-decisions.json',dict(status='explicitly_reviewed',decisions=all_decisions,inputs=inputs))
    audit('final');admit('final')

if __name__=='__main__':main()
