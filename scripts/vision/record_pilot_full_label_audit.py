"""Record explicit observations made from the twelve evidence pages, not plan themes."""
from datetime import datetime, timezone
from pathlib import Path
import re
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision.build_pilot_audit_pages import BASE

# Per-box observations in receipt order. U means unresolved visible content,
# not a claim that the instance is absent. V is visual content, not admission.
OBS = [
 [('V','上部箱体及顶端结构可见，杆体和前景遮挡下部'),('V','图缘截断，箱体上部和顶端可见'),('U','裁剪主要为重叠青色带状结构，目标归属不清'),('U','裁剪主要为前景块体，后方柜体可辨内容无法确认'),('V','圆柱主体及右侧基座可辨，左下遮挡')],
 [('V','右侧箱体及顶端可辨，右缘截断'),('V','圆柱主体和基座清晰')],
 [('V','箱体上部及顶端可见，杆体遮挡'),('V','箱体上部及顶端可见，下部遮挡'),('U','主要为重叠横带，目标内容无法确认'),('U','主要为前景块体，后方目标内容无法确认'),('U','左图缘极窄竖片，类别内容不足'),('V','圆柱及基座可辨，左下遮挡'),('U','窄横带重叠，目标内容无法确认'),('U','主要为电抗器遮挡，右侧小片段内容不足')],
 [('V','箱体上部及顶端可见，杆体及前景遮挡'),('U','框内主要为前景青色块体'),('V','上半圆柱清楚，下半被前景大幅遮挡'),('U','框内主要为前景及电抗器，无法分辨目标')],
 [('V','圆柱左侧和基座可辨，右侧大幅遮挡；遮挡物身份尚需空间核验')],
 [('V','箱体上部及顶端可辨，下部遮挡'),('U','主要为前景块体，目标内容无法确认'),('V','圆柱上部右侧及部分基座可辨，左下遮挡')],
 [('V','箱体与顶端结构可辨，左缘接触'),('V','上部和顶端可辨，前景及杆体遮挡'),('V','远处箱体上部与顶端可辨'),('U','重叠横带，目标可辨内容无法确认'),('U','前景块体占据主要区域'),('U','主要为近处变压器和前景块体'),('V','圆柱及部分基座可辨，基座右缘截断'),('U','很小的重叠带状片段')],
 [('V','圆柱清晰，基座右侧被图缘截断')],
 [('U','框内主要为前景块体及电抗器'),('V','圆柱及基座可辨，左下遮挡并接触左图缘')],
 [('V','远处箱体上部与顶端可见，下部遮挡'),('V','柜状主体和基座可辨'),('V','远处圆柱和基座可辨'),('V','近处块体主体和基座可辨，右缘截断；类别依赖来源映射')],
 [('V','箱体上部和顶端可辨，前景遮挡'),('U','极窄横框内容为前景块体及杆体，目标内容不足'),('U','右缘极窄竖片，目标内容不足'),('V','小圆柱及基座可辨'),('U','远处狭窄块体片段，内容和归属不清'),('V','近处主体与基座可辨，杆体遮挡；类别依赖来源映射')],
 [('U','变压器框几乎全部为前景电容器组表面，无法确认可辨内容'),('V','柜状主体左侧与基座可辨，右侧遮挡'),('V','远处圆柱及基座可辨'),('V','近处块体及基座可辨，右缘截断；类别依赖来源映射')],
]

def require_clear_review(record):
    if record.get('status') != 'complete_clear_for_bounded_training':
        raise ValueError('Full-label semantic review is not clear')
    rows=record.get('boxes',[])
    ids=[(r['view_id'],r['annotation_id']) for r in rows]
    if not rows or len(ids)!=len(set(ids)) or any(r['state']!='visible_content' for r in rows):
        raise ValueError('Missing, duplicate or unresolved box review')
    for p,h in record['inputs'].items():
        if file_sha256(p)!=h:raise ValueError('Stale review evidence')

def run():
    root=BASE/'pilot-full-label-audit-v1'
    receipt=BASE/'pilot/collection-receipt.json'; plan_path=BASE/'plan.json'
    plan=read_record(plan_path); frames=read_record(receipt)['views']
    evidence=read_record(root/'evidence.json'); by_view={r['view_id']:r for r in evidence['rows']}
    mapping={}
    for o in plan['objects']:
        for label in set(o['runtime_labels']):
            if int(label) in mapping:raise ValueError('Instance collision')
            mapping[int(label)]=o
    deps={str(p.resolve()):file_sha256(p) for p in (receipt,plan_path,root/'evidence.json',Path(__file__))}
    boxes=[]
    assert len(frames)==len(OBS)==12
    for v,obs in zip(frames,OBS):
        e=by_view[v['view_id']]
        for p,h in ((e['page'],e['page_sha256']),(v['rgb_path'],v['image_sha256'])):
            if file_sha256(p)!=h:raise ValueError('Changed viewed evidence')
            deps[p]=h
        assert len(obs)==len(v['truth']['objects'])
        for o,(state,reason) in zip(v['truth']['objects'],obs):
            label=int(re.search(r'instance-(\d+)',o['annotation_id'])[1]); source=mapping[label]
            if source['category']!=o['class_name']:raise ValueError('Category mismatch')
            boxes.append(dict(view_id=v['view_id'],annotation_id=o['annotation_id'],scene_device_id=source['name'],bbox_xyxy=o['bbox_xyxy'],state='visible_content' if state=='V' else 'unresolved_content',reason=reason,pixel_visibility_certified=False))
    record=dict(status='reviewed_with_named_gaps',review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),boxes=boxes,inputs=deps,training_admitted=False,promotable=False,
                limitation='Visual content judgments are not pixel-level instance certification. No training approval; unresolved boxes block the unchanged schedule.')
    out=root/'full-label-review.json'
    write_record(out,record)
    print('Reviewed',len(boxes),'unresolved',sum(r['state']=='unresolved_content' for r in boxes))

if __name__=='__main__':run()
