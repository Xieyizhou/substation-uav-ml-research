"""Import explicit AI visual observations made on this run's 49 evidence pages."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.evaluate_unified_lighting import OUT,prior

# Explicit observations, not model-generated approval or inherited historical reviews.
FP={
 'FP01':[('block','灰色大块体侧面、深色面板及基座；右缘包含细杆。')],
 'FP02':[('block','灰色块体的大面积侧面与右侧深色面板，底部基座清楚。')],
 'FP03':[('cabinet_like','青色柜状块体，正面深色矩形面板、顶面及底座完整。')],
 'FP04':[('mixed','后方灰色块体与前景粗杆、右下青色块体共同落框。'),('mixed','灰色块体侧面被两根前景杆体遮挡，右下含青色块体。'),('mixed','同一后方灰块与前景杆体重叠结构，不能归为纯杆体。')],
 'FP05':[('mixed','主要为灰色块体侧面；右侧粗杆及前景青色块体侵入。')],
 'FP06':[('ground_shadow','框内主要是地面网格与横向杆影；上边缘只含后方块体基座。'),('block','灰色块体正面深色矩形面板与底座，不是独立杆体。')],
 'FP07':[('block','灰色块体正面面板与底座，被预测为电容器组。'),('block','同一灰色块体正面，被该seed同时预测为电抗器。'),('block','灰色块体正面及底座，重复覆盖前两框所见结构。'),('block','同一灰块面板被另一组预测为电容器组。')],
 'FP08':[('edge_block_ground','右图缘灰色块体局部、面板和基座，框下半部含大量地面。'),('edge_block_ground','同一右缘灰块局部与地面组合，被预测为电抗器。')],
}
LOSS={
 1:'电容器组块体顶面及宽侧面可见；下部被前方青色柜体遮挡，面板不见。',
 2:'电抗器圆柱主体及底座清晰，未见显著遮挡，画内目标并不微小。',
 3:'多台柜体前后叠置，只能分辨顶面及细窄侧面条带，目标像素归属不足。',
 4:'后方电容器块体顶面及侧面可见，下部被青色柜体及前景变压器遮挡；框内端子属前景，不能当作目标特征。',
 5:'柜体侧面、顶面、正面面板一角及底座清楚，未见明显遮挡。',
 6:'远处柜体顶面、右侧及面板窄条可见，左下被邻柜与前景端子遮挡。',
 7:'柜体顶面及面板上部可辨，下部被前景变压器遮挡。',
 8:'变压器宽顶面、三个端子及侧面清楚，下部出画；截断但内容丰富。',
 9:'左缘变压器宽侧面与顶部端子可见，前景粗杆遮挡中央，左侧出画。',
 10:'远处柜体顶面和侧背面局部可见，下左部分被前方柜体遮挡。',
 11:'中景柜体顶面与背面可见，下部被前景变压器遮挡，邻柜相近。',
 12:'右缘远处多柜前后重叠，框内包含多层顶面和窄侧面，无法可靠分离目标像素。',
 13:'右缘多柜重叠且出画，目标只呈局部窄侧面，像素归属待核。',
 14:'右缘柜体侧面与底座部分可辨，出画且邻柜遮挡，非完整外形。',
 15:'后方变压器顶面与三个端子可见，较低位置被前景变压器遮挡。',
 16:'原始条件右缘远处叠置柜体，只见顶面与侧面条带，目标像素归属待核。',
 17:'原始条件右缘柜体重叠、出画，不能把框中所有蓝色区域归给同一实例。',
 18:'原始条件右缘柜体的侧面与部分基座可见，画缘与邻柜限制完整外形。',
 19:'近景电抗器完整圆柱、圆顶和底座可辨；无明显前景遮挡。',
 20:'左缘仅极窄上部块体条带，下方为前景柜体，无法从RGB认证该变压器实例像素。',
 21:'电容器组宽背侧面与底座清晰，表面存在斜向明暗变化，无明显遮挡。',
 22:'中远景变压器主体、顶面端子及底座可辨，右侧被细杆遮挡。',
 23:'右侧远处多柜顶面及窄背面叠置，目标与邻柜像素分界不足。',
 24:'远景柜体顶部与右侧局部可见，左半被变压器遮挡；尺度较小。',
 25:'中景柜体宽顶面与背侧面可辨，下部被近景柜体遮挡，面板不在视角内。',
 26:'近景柜体顶面和宽背面可见，下部出画；框顶含邻柜局部。',
 27:'原始条件变压器主体、顶面三个端子及底座可辨，右侧前景杆遮挡。',
 28:'变压器主体、三个端子及基座清晰，右后方有小柜体但主要目标可分辨。',
 29:'原始条件右侧远处多柜叠置，仅顶面及侧面条带，目标像素归属待核。',
 30:'原始条件中景柜体宽顶面、背侧面可见，下部被近景柜体遮挡。',
 31:'原始条件近景柜体顶面与背侧面清楚，下部出画。',
 32:'远处电容器组上半块体可见，下部被变压器遮挡；框内三个端子属前景，不能归给目标。',
 33:'孤立柜体完整顶面、两侧及底座可辨，无面板朝向；并非仅有微小片段。',
 34:'原始条件孤立柜体宽顶面、侧背面及基座清晰，未见明显遮挡。',
 35:'变压器宽主体与顶部端子可见，中央前景杆体遮挡较大条带。',
 36:'电容器组顶面与主体大部可见，左下被前景小柜体遮挡。',
 37:'左缘柜体宽侧面及底座可辨，但部分外形出画，面板不见。',
 38:'左缘后方柜体仅上部条带，前景柜体遮挡且出画，实例像素归属不足。',
 39:'后方电抗器圆顶和上部圆柱清楚，下半被变压器遮挡；前景端子不属于电抗器。',
 40:'原始条件左缘后方柜体仅上部条带，不能把前景柜体像素计为目标可见内容。',
 41:'原始条件后方电抗器圆顶与上部圆柱可辨，下半被变压器遮挡。',
}
PENDING={3,12,13,16,17,20,23,29,38,40}

def validate(e,ds):
    expected={}
    for event in e['events']:
        for j in range(len(event['predictions']) if event['kind']=='FP' else 1): expected[f"{event['event_id']}:{j}"]=(event,j)
    if len(ds)!=len(expected) or {d['decision_id'] for d in ds}!=set(expected):raise ValueError('Missing/duplicate review')
    for d in ds:
        event,j=expected[d['decision_id']]
        if d['evidence_sha256']!=event['page_sha256'] or d['image_sha256']!=event['source']['image_sha256'] or d['crop_sha256']!=event['crops'][j]['sha256']:raise ValueError('Stale review evidence')
        if not d['reason'] or not d['reviewed_at'] or d['review_nature']!='AI-assisted' or d['status'] not in ('reviewed','pending'):raise ValueError('Invalid review')

def main():
    ep=OUT/'error-review/evidence.json';e=prior.read(ep);prior.verify(e);ds=[];time=datetime.now(timezone.utc).isoformat()
    for event in e['events']:
        eid=event['event_id'];isfp=event['kind']=='FP';items=FP[eid] if isfp else [('target_content',LOSS[int(eid[4:])])]
        if len(items)!=(len(event['predictions']) if isfp else 1):raise ValueError('Explicit observation coverage mismatch')
        for j,(kind,note) in enumerate(items):
            ds.append(dict(decision_id=f'{eid}:{j}',status='pending' if not isfp and int(eid[4:]) in PENDING else 'reviewed',
                content_category=kind,reason=note,review_nature='AI-assisted',reviewed_at=time,
                evidence_sha256=event['page_sha256'],image_sha256=event['source']['image_sha256'],crop_sha256=event['crops'][j]['sha256'],
                pixel_visibility_certified=False,asset_identity='not_reclassified_from_appearance',
                prediction=event['predictions'][j] if isfp else None,truth=event['loss']['truth'] if not isfp else None))
    validate(e,ds)
    prior.frozen(OUT/'error-review/review.json',dict(status='explicit_review_complete_with_named_gaps',decisions=ds,
        pending_ids=[d['decision_id'] for d in ds if d['status']=='pending'],
        scope='RGB full-frame and crop observations; no masks, label revisions, or training admission. Operational miss types retained in evidence.',
        inputs={str(x):prior.file_sha256(x) for x in (ep,Path(__file__).resolve())}))
    print('REVIEWED',len(ds),'PENDING',sum(d['status']=='pending' for d in ds))

if __name__=='__main__':main()
