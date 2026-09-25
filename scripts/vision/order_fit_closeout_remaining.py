"""Explicit evidence adjudication; never manufacture decisions from summary counts."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision import import_transfer_pilot_review as pilot
from scripts.vision import import_transfer_expansion_review as expansion

LEGACY = {
 'C02':'后变压器上部及附件、受杆遮挡的圆柱和完整电容器外壳可辨，保留遮挡限制。',
 'C03':'两台变压器仍有主体与附件，前后遮挡单独保留；电抗器完整。',
 'C04':'边缘变压器仍有宽主体、顶面及附件片段，另一台和电容器完整；记录截断。',
 'C05':'封闭箱体顶面、宽侧面和基座完整，只支持封闭外壳视图。',
 'C06':'变压器主体三端子及柜体面板、侧面和基座可辨，局部下角遮挡保留。',
 'C09':'后变压器上部和三个端子、其他变压器主体附件可辨；长柜侧面基座和电抗器可辨，保留后排遮挡。',
 'C10':'变压器主体多端子、柜体宽侧面与基座保留；分别记录图缘和下角遮挡。',
 'C12':'变压器顶面主体及重叠附件可辨，长柜和箱体基座可见；不宣称正面板可见。',
 'C13':'变压器主体三端子清楚，柜体有宽侧面及基座；后方左侧遮挡保留。',
 'C15':'右缘变压器仍有宽主体和附件，电抗器及基座清楚，保留截断。',
 'C17':'主体端子与完整箱体可辨，杆体遮挡分别记录，不宣称完整可见。',
 'C18':'近变压器宽主体及三个端子仍可辨，另一台和箱体保留主体；底缘截断及杆遮挡保留。',
 'C20':'后柜仍有面板、近灰柜有宽侧面顶面基座，变压器与面板柜清楚，圆柱下右遮挡保留。',
 'C24':'俯视电抗器主体及基座完整可辨。',
 'C27':'两台变压器保留宽主体顶面与附件，箱体宽侧面基座可辨；记录双侧截断。',
 'C28':'完整变压器与边缘变压器的宽主体附件可辨，柜体宽侧面基座保留。',
 'C29':'变压器主体和附件保留，后柜宽侧面受前景遮挡；作为有限内容视图，不认证全可见。',
 'C35':'完整箱体顶面、侧面和基座可辨。',
 'C36':'后变压器上部主体和三端子仍可辨，箱体和基座清楚，杆及前景遮挡保留。',
 'C39':'各目标仍有宽主体、侧顶面或端子依据，近变压器底截断及多处杆遮挡保留；圆柱大弧面可辨。',
 'N01':'独立箱体顶面、侧面及基座完整，身份依赖保存来源，不将外壳当独立类别证据。',
 'N03':'变压器主体三端子仍清楚，另一箱体宽主体顶面及基座完整。',
 'N04':'图缘变压器主体多端子及基座保留，各柜宽主体与基座可辨，记录前景遮挡。',
 'N06':'后面板局部、宽侧面顶面及基座可辨，变压器和面板柜完整，圆柱下右遮挡保留。',
 'N08':'分别保留灰蓝柜和后柜的遮挡，完整变压器、面板柜及圆柱宽主体可辨。',
 'N09':'变压器宽主体、顶面三端子和基座完整可辨。',
}
GRAY = {
 'G01':'电容器封闭主体和底座清楚，变压器主体与三端子完整；不把封闭外壳称为内部结构。',
 'G02':'逐柜顶面、侧壁或保留面板可辨，后排下部和近柜右下遮挡保留；变压器三端子主体完整。',
 'G03':'圆柱顶面、侧壁和黑色基座完整，灰色渐变仍可辨。',
 'G04':'变压器主体端子完整，后圆柱顶面及较大主体保留，下右遮挡不视作完整可见。',
 'S01':'电容器封闭外壳和变压器主体端子均完整可辨。',
 'S02':'两台变压器主体端子清楚，后电容器较小但轮廓、顶前侧和基座可辨。',
 'S03':'长柜顶面、端侧与基座清楚，变压器和入口柜可辨；当前视角不认证正面板。',
 'S04':'逐柜顶背侧、保留面板或基座可辨，后柜下角遮挡保留；变压器三端子主体完整。',
 'S05':'灰色圆柱及基座完整，右侧杆体不作为目标内容。',
 'S06':'灰色圆柱椭圆顶面、大侧壁和基座可辨，地面阴影不等于遮挡。',
 'S07':'变压器完整箱体端子清楚，后电抗器宽主体保留但右下基座被遮挡。',
 'S08':'变压器完整主体三端子与电容器完整外壳可辨，近旁杆体分离。',
}
RISK = {
 'R03':(False,'后柜保留面板和主体，左近柜虽截断仍有宽顶侧面及基座，变压器多端子与圆柱可辨；接受封闭侧面限制，不认证全可见。'),
 'R14':(True,'左图缘柜仅侧面基座，缺少完整外轮廓；当前证据不足以解决历史内容风险，整图暂缓。'),
 'R22':(True,'右图缘变压器只有侧壁基座片段，无顶部附件或完整主体；整图暂缓，不用清晰电抗器替代。'),
 'R25':(True,'右图缘变压器仅侧壁窄顶面和底座，主体附件不完整；保留历史风险并整图暂缓。'),
}


def validate_decisions(rows, expected, pool):
    ids=[d['member_id'] for d in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('Missing, duplicate or unexpected adjudication')
    for d in rows:
        m=pool[d['member_id']]
        if not d['reason'] or not d['retained_label_decisions']:
            raise ValueError('Missing authored reason or full-label evidence')
        if len(d['retained_label_decisions'])!=len(m['truth']):
            raise ValueError('Incomplete label decisions')
        for key in ('image','label'):
            if d[key+'_sha256']!=m[key+'_sha256']:
                raise ValueError('Stale decision identity')
        if d['quality_status'] not in ('supported_with_recorded_limits','whole_frame_held'):
            raise ValueError('Unknown quality decision')


def run():
    dest=OUT/'closeout-remaining-quality.json'
    if dest.exists():
        record=prior.read(dest);prior.verify(record);return record
    paths=[]
    def read(name):
        path=OUT/name;record=prior.read(path);prior.verify(record);paths.append(path);return record
    p=read('protocol.json');inventory=read('closeout-inventory-v1.json')
    reused=read('closeout-source-variant-quality.json')
    legacy=read('legacy-review-links.json');resolution=read('source-resolution.json')
    read('remaining-variant-review-links.json');read('gray-full-label-correspondence.json')
    early=read('early-positive-review-links.json')
    risk=read('risk-reconciliation/review.json');read('risk-reconciliation/evidence.json')
    pool={m['member_id']:m for m in p['members']}
    expected={m['member_id'] for m in inventory['members'] if m['stage']=='final_quality_disposition_required'}-{m['member_id'] for m in reused['members']}
    rows=[]
    def add(mid,reason,decisions,held=False):
        m=pool[mid]
        for key in ('image','label'):
            path=Path(m[key+'_path'])
            if prior.file_sha256(path)!=m[key+'_sha256']:raise ValueError('Current member drift')
            paths.append(path)
        rows.append(dict(member_id=mid,reason=reason,quality_status='whole_frame_held' if held else 'supported_with_recorded_limits',
            retained_label_decisions=decisions,image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            evidence_basis='Adjudication of existing signed per-label observations; not a new mask certification.',
            pixel_visibility_certified=False,training_eligible=False,
            remaining_gates=['full_scene_source_role_confirmation','risk_derivation_closure','export_and_loader_preflight']))
    for m in legacy['members']:
        if m['member_id'] not in expected:continue
        rid=m['decisions'][0]['event_id'].split('-')[0]
        if rid=='N02':
            reason='重新查看全图：近右变压器大幅截断且仅一根附件；左前未框箱体还需与来源逐一对应。整图暂缓，不断言漏标或自动批准。'
            add(m['member_id'],reason,m['decisions'],True)
        else:add(m['member_id'],LEGACY[rid],m['decisions'])
    for mod in (pilot,expansion):
        ep=mod.DEST/'evidence.json';rp=mod.DEST/'label-review.json'
        e=prior.read(ep);r=prior.read(rp);prior.verify(e);prior.verify(r)
        mod.validate(e,r['decisions']);paths.extend([ep,rp,Path(mod.__file__).resolve()])
        ds={d['event_id']:d for d in r['decisions']}
        for page in e['pages']:
            if page['condition'] not in ('gray_target_body','gray_all_body'):continue
            mid=page['source_id']+'-'+page['condition']
            events=[v for v in e['events'] if v['source_id']==page['source_id'] and v['condition']==page['condition']]
            if len({v['object_id'] for v in events})!=len(events):raise ValueError('Instance collision')
            add(mid,GRAY[page['source_id']]+' 本决定仅适用于已绑定的'+page['condition']+'逐标签证据。',[ds[v['event_id']] for v in events])
    for m in resolution['resolved_sources']:
        add(m['member_id'],'复用本帧 accepted_with_recorded_limits 的完整标签审核；保留后排或图缘遮挡，不继承原图像素认证。',m['review_decisions'])
    for rid,(held,reason) in RISK.items():
        decisions=[d for d in risk['decisions'] if d['review_id']==rid]
        add(decisions[0]['member_id'],reason,decisions,held)
    validate_decisions(rows,expected,pool)
    if len(rows)!=59:raise ValueError('Expected exactly 59 remaining dispositions')
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='remaining_quality_adjudicated_not_dataset_admission',decisions=rows,
        counts=dict(Counter(d['quality_status'] for d in rows)),
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['counts'])
