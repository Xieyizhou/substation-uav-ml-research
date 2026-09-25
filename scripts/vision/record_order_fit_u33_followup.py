"""Preserve uncertainty while adding the viewed component evidence."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    ep=OUT/'u33-component-projection.json';rp=OUT/'u33-fragment-review.json';tp=OUT/'legacy-world-source-trace.json'
    e,r,t=[prior.read(p) for p in (ep,rp,tp)]
    for record in (e,r,t):prior.verify(record)
    if e['member_id']!=r['member_id']:raise ValueError('Member mismatch')
    source=next(x for x in t['members'] if x['member_id']==e['member_id'])
    pp=Path(source['source_plan']);op=pp.parent/'obstacles.json';page=Path(e['page_path'])
    if prior.read(pp)['files']['obstacles.json']!=prior.file_sha256(op):raise ValueError('Source taxonomy drift')
    assets=[x for x in prior.read(op)['obstacles'] if x['name']=='cabinet_2']
    if len(assets)!=1 or assets[0]['visual_category']!='cabinet':raise ValueError('Background identity differs')
    dest=OUT/'u33-fragment-followup.json'
    if dest.exists():
        result=prior.read(dest);prior.verify(result);return result
    deps=[ep,rp,tp,pp,op,page,Path(__file__).resolve()]
    return prior.frozen(dest,dict(status='ordinary_cabinet_base_association_supported_with_limits',
        member_id=r['member_id'],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
        reason='已查看组件投影叠图，左缘窄深色片段位于cabinet_2基座投影区域；保存世界定义该资产为普通柜体，主体及面板投影在画外。支持背景基座解释，但投影包围框明显大于可见片段，不能作为精确像素证据。',
        previous_unknown_record=str(rp),pixel_visibility_certified=False,training_eligible=False,
        full_frame_eligibility='pending_explicit_final_disposition',
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__':print(run()['status'])
