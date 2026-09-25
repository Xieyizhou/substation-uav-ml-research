"""Explicit observations of the frozen 49 + 148 targets; never admits or trains."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from scripts.vision.condition_review_evidence import OUT, VARIANTS
from scripts.vision.condition_transfer_design import ROOT, read, verify, frozen, file_sha256
from src.vision.canonical.plan import read_record

EVIDENCE_ID = 'f2b618ecd8360137670f605fa123ffd9ebaffcc66b3b622895f4fb29681c96c0'
PANEL = {'N':'not_visible_on_exposed_faces','L':'visible_low_contrast',
         'D':'visible_distinct_contrast','U':'unknown'}
OCC = {'N':'no_obvious_foreground_occlusion','P':'partial_foreground_occlusion',
       'H':'heavy_foreground_occlusion','U':'unknown'}
# These rows transcribe inspection of all nine training pages, ten four-condition
# crop pages and four full-context pages. No model outcome determines a code.
TRAIN = [
 ('UU','右图缘仅余窄条和基座片段，无法可靠判断面板或归属。'),
 ('NN','完整普通箱体正面、顶部和基座，未见面板。'),
 ('NN','普通箱体正侧面、顶部及基座可见；后方变压器未明显遮住主体。'),
 ('NN','远处小箱体顶部、平面和基座可见。'),
 ('NN','高视角可见顶部、普通平面及基座。'),
 ('NP','近处青色箱体遮住下部正面，仍可见上部和右侧平面。'),
 ('UU','左图缘仅余极窄片段，地面与基座区域不能认证为主体像素。'),
 ('NN','普通正侧面、顶部及基座可见。'),
 ('NN','普通正面、顶部及基座可见。'),
 ('NN','远处普通正侧面及基座可见。'),
 ('NN','远处普通箱体正侧面、顶部及基座可见。'),
 ('NN','画面左部可见普通正侧面及基座。'),
 ('NN','大面积普通平面及基座带斜向阴影；保持既有零曝光，不据此重新准入。'),
 ('NN','普通正侧面、顶部及基座可见。'),
 ('NP','近处青色箱体遮住下部正面，上部平面和顶部可见。'),
 ('NN','普通箱体平面、顶部及基座可见，无明显前景重叠。'),
 ('NP','前景变压器遮住右下部，后方上部普通平面可见。'),
 ('NN','高视角顶部、普通箱体平面及基座可见。'),
 ('NN','左下图缘截断，仍可见宽阔普通平面、顶部和部分基座。'),
 ('NN','改变背景的普通正面及基座带斜向阴影，未见面板。'),
 ('NP','近处青色箱体遮住左下部，后方上部与右侧可见。'),
 ('LP','后方灰色柜体有低对比面板轮廓，前方柜体遮住部分下部。'),
 ('LN','近处灰色柜体面板轮廓可见且对比弱，宽侧面、顶部及基座可见。'),
 ('NP','改变背景后，仍由近处青色箱体遮住后方目标左下部。'),
 ('LH','后方灰色柜体面板上缘可辨，前方柜体遮住较多下部和左部。'),
 ('LP','近处灰色柜体面板边界对比弱，左下图缘截断并有底部前景片段。'),
 ('LN','灰色面板轮廓、顶部及基座可见，无明显前景遮挡。'),
 ('NP','灰色普通箱体上部和顶部可见，左下被青色箱体遮住。'),
 ('NN','灰色普通平面及基座有斜向阴影，未见面板。'),
 ('NN','青色普通正面及基座有斜向阴影，未见面板。'),
 ('UH','前景变压器屋顶和套管占据大部分框，后方仅余上部窄条；套管不归给电容器。'),
 ('UU','右图缘极窄箱体及基座片段，不足以可靠判断面板。'),
 ('NP','近处青色箱体遮住左下部，后方普通平面可见。'),
 ('NP','前景变压器遮住右下部，后方普通平面和顶部可见。'),
 ('NH','前景变压器与青色箱体重叠遮住较多内容，仅余后方上部及右部平面。'),
 ('NN','近处普通平面、顶部及基座可见。'),
 ('NN','近处高视角顶部及普通平面可见，底部图缘截断。'),
 ('NP','后方普通平面、顶部及基座部分被右侧变压器遮住。'),
 ('UH','后方小目标夹在前景变压器之间，仅余窄顶部和平面，不足以可靠归属面板。'),
 ('NP','左部被前景变压器遮住，右部普通平面、阴影及基座可见；保持既有零曝光。'),
 ('NN','普通平面、顶部及基座带斜向阴影；保持既有零曝光。'),
 ('NN','普通正面及基座带阴影，未见面板。'),
 ('UH','框内主要为前景变压器屋顶及主体，后方电容器仅余窄条，内容归属不足。'),
 ('NN','近处普通平面、顶部及基座可见。'),
 ('NP','宽阔普通平面及基座可见，前景杆体与右侧边缘重叠。'),
 ('NN','普通平面、顶部及基座带斜向阴影。'),
 ('UH','后方蓝色柜体上部被前方柜体遮住，面板无法可靠归属，保留既有待定。'),
 ('UH','前景变压器屋顶和套管占主要区域，后方电容器仅余上部窄条。'),
 ('NN','普通正面及基座可见；邻近杆体在后方，未见明显遮挡。'),
]
# Panel order: original, material, background, lighting. Every column was viewed.
DEV = [
 ('NNNN','N','完整普通后侧面、顶部及基座，未显露前面板。'),
 ('NNNN','N','完整普通箱体平面、顶部及基座。'),
 ('NNNN','H','后方目标上部和顶部可见，下部被前方成列箱体遮住。'),
 ('NNNN','P','左部被变压器遮住，右侧普通平面和基座可见。'),
 ('NNNN','P','后方上部及顶部可见，下部被近处箱体遮住。'),
 ('NNNN','P','后方上部及顶部可见，下部被近处箱体遮住。'),
 ('NNNN','N','近处普通平面及顶部可见，右下图缘截断。'),
 ('UUUU','H','前景变压器屋顶和套管占主要内容，后方电容器仅余上部窄条。'),
 ('NNNN','P','上部、顶部及右侧基座可见，下部被近处青色箱体遮住。'),
 ('NNNN','N','左图缘截断的普通平面、顶部及基座可见。'),
 ('NNNN','P','后方上部及顶部可见，下部被近处箱体遮住，左图缘截断。'),
 ('NNNN','N','宽阔普通平面、部分顶部和基座可见，左下图缘截断。'),
 ('NNNN','P','普通平面、顶部及基座可见，左下被近处青色箱体遮住。'),
 ('NNNN','N','完整普通正面、顶部及基座，未显露前面板。'),
 ('NNNN','P','斜向成列箱体中后方目标顶部及普通窄平面可见，前景遮住下部。'),
 ('UUUU','H','前景变压器侧面占主要区域，后方开关柜仅余侧面和基座窄片段。'),
 ('NNNN','P','成列箱体中普通平面及顶部局部可见，有前景遮挡。'),
 ('NNNN','P','窄长普通侧面及基座可见，右侧图缘截断并有相邻结构重叠。'),
 ('UUUU','U','右图缘仅余极窄片段和地面/基座，无法可靠判断面板。'),
 ('NNNN','H','后方普通平面及顶部可见，杆体横跨左部，变压器遮住右下部。'),
 ('NNNN','N','大面积普通正面及基座带斜向阴影，杆体邻近但未明显遮住主体。'),
 ('DLDD','P','前面板可辨，下部被近处柜顶部分遮住。'),
 ('DLDD','N','前面板上部可辨，目标被底部图缘截断，顶部和宽阔箱体可见。'),
 ('UUUU','H','前景箱体上方仅余很薄的疑似面板上缘，面积不足以判断面板对比。'),
 ('UUUU','H','多个成列箱体顶部占主要区域，面板仅疑似窄边，不足以可靠判断对比。'),
 ('NNNN','P','后方普通平面及顶部可见，变压器屋顶覆盖底部窄区域。'),
 ('NNNN','P','普通平面、顶部及基座可见，左部被前景变压器遮住。'),
 ('NNNN','N','左图缘截断，仍有宽阔普通平面及顶部片段。'),
 ('NNNN','N','完整普通正侧面、顶部及基座可见。'),
 ('NNNN','P','近处箱体遮住右下部，上部普通平面、顶部及左侧基座可见。'),
 ('NNNN','P','近处箱体遮住左下部，后方普通平面及顶部可见。'),
 ('NNNN','H','电容器上部及顶部可见，下部被青色箱体、变压器屋顶和套管遮住；不把套管归给电容器。'),
 ('DLDD','P','宽面板可辨，下部被变压器屋顶遮住；右侧套管属于前景。'),
 ('DLDD','N','宽阔右侧面、窄左面板、顶部及基座均可见。'),
 ('DLDD','P','面板、侧面、顶部及基座可辨，底部有少量前景角部。'),
 ('DLDD','P','面板左部被前方箱体遮住，右部箱体、顶部及基座可见。'),
 ('DLDD','H','后方面板及主体被近处箱体和套管部分遮住，剩余面板条带及边界仍可辨。'),
]


def binding(row):
    s = row['source']
    b = dict(image_sha256=s['image_sha256'], evidence_sha256=row['evidence_sha256'],
             truth=row['target']['truth'])
    for k in ('label_sha256','truth_sha256'):
        if k in s: b[k] = s[k]
    if 'full_context_sha256' in row: b['full_context_sha256'] = row['full_context_sha256']
    return b


def validate(evidence, decisions, check_files=True):
    rows = evidence['training'] + evidence['development']
    expected = {r['review_id']:r for r in rows}
    if len(expected)!=len(rows) or len(decisions)!=len(rows) or {d['review_id'] for d in decisions}!=set(expected):
        raise ValueError('Missing or duplicate review')
    for d in decisions:
        r = expected[d['review_id']]
        if d['binding'] != binding(r): raise ValueError('Stale review binding')
        if d['panel_condition'] not in PANEL.values() or d['occlusion_condition'] not in OCC.values():
            raise ValueError('Unknown condition enum')
        held = 'unknown' in (d['panel_condition'],d['occlusion_condition'])
        if d['decision'] != ('held_condition_unknown' if held else 'condition_recorded'):
            raise ValueError('Unknown silently accepted')
        if d['pixel_visibility_certified'] or d['training_admitted'] or d['promotable']:
            raise ValueError('Review cannot certify visibility or admission')
        if d['review_nature']!='AI-assisted' or not d['reason'] or not d['reviewed_at'] or not d['full_context_checked']:
            raise ValueError('Incomplete explicit review')
        if check_files:
            for obj, kinds in ((r['source'],('image','label')),(r,('evidence','full_context'))):
                for k in kinds:
                    if k+'_path' in obj and file_sha256(obj[k+'_path'])!=obj[k+'_sha256']:
                        raise ValueError('Stale evidence file')


def decisions(e):
    if e['identity']!=EVIDENCE_ID or len(e['training'])!=49 or len(e['development'])!=148:
        raise ValueError('Observations belong to another frozen target set')
    notes={f'T{i:03}':(code[0],code[1],why) for i,(code,why) in enumerate(TRAIN,1)}
    for i,(codes,occ,why) in enumerate(DEV,1):
        for v,code in zip(VARIANTS,codes):
            contrast = (' 当前列面板和外壳接近灰色，对比较弱。' if code=='L' else
                        ' 当前列深色面板与外壳有可辨对比。' if code=='D' else '')
            notes[f'D{i:03}-{v}']=(code,occ,why+contrast)
    ds=[]
    for row in e['training']+e['development']:
        p,o,reason=notes[row['review_id']]
        ds.append(dict(review_id=row['review_id'],binding=binding(row),panel_condition=PANEL[p],
            occlusion_condition=OCC[o],reason=reason,decision='held_condition_unknown' if 'U' in (p,o) else 'condition_recorded',
            full_context_checked=True,review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
            pixel_visibility_certified=False,training_admitted=False,promotable=False,
            component_identity='not_inferred_from_visual_appearance'))
    return ds


def component_audit(e):
    """Saved development worlds, receipt mapping and each original annotation ID."""
    out=[];paths=[]
    for variant in VARIANTS:
        group=[r for r in e['development'] if r['target']['variant']==variant]
        captures={Path(r['source']['image_path']).parent.parent for r in group}
        for capture in sorted(captures):
            wp=capture.parent/'plan/world.sdf';pp=capture.parent/'plan/plan.json';cp=capture/'collection-receipt.json'
            plan=read_record(pp);receipt=read_record(cp);digest=file_sha256(wp)
            if digest!=plan['files']['world.sdf'] or digest!=receipt['world_sha256'] or receipt['plan_identity']!=plan['identity']:
                raise ValueError('Saved world/receipt/plan mismatch')
            raw_mapping=receipt['collection_checks']['instance_mapping']
            mapping=normalize_mapping(raw_mapping)
            if len({v['object_id'] for v in mapping.values()})!=len(mapping):raise ValueError('Mapping collision')
            models={m.get('name'):m for m in ET.parse(wp).findall('.//world/model')}
            targets=[]
            for r in group:
                if Path(r['source']['image_path']).parent.parent!=capture:continue
                t=r['target']['truth'];match=re.search(r'-instance-(\d+)-',t['annotation_id'])
                if not match or str(int(match[1])) not in mapping:raise ValueError('Unresolved annotation instance')
                identity=mapping[str(int(match[1]))]
                if identity['category']!=t['class_name']:raise ValueError('Class mapping conflict')
                m=models[identity['object_id']];visuals=[v.get('name') for v in m.findall('.//visual')]
                targets.append(dict(review_id=r['review_id'],instance_label=match[1],**identity,
                    visual_names=visuals,front_panel_present='front_panel' in visuals))
            out.append(dict(variant=variant,world_sha256=digest,world_path=str(wp),
                capture_path=str(cp),targets=targets))
            paths += [wp,pp,cp]
    return out,paths


def normalize_mapping(mapping):
    result={}
    for key,value in mapping.items():
        if not str(key).isdigit():raise ValueError('Non-numeric instance mapping')
        label=str(int(key))
        if label in result:raise ValueError('Normalized instance mapping collision')
        result[label]=value
    return result


def summarize(e, ds):
    lookup={d['review_id']:d for d in ds};result={}
    for role in ('training','development'):
        result[role]=dict(Counter(lookup[r['review_id']]['panel_condition'] for r in e[role]))
    misses={}
    for v in VARIANTS:
        misses[v]={}
        for cls in ('switchgear','capacitor_bank'):
            rows=[r for r in e['development'] if r['target']['variant']==v and r['target']['truth']['class_name']==cls]
            misses[v][cls]=dict(total=len(rows),persistent_misses_by_panel=dict(Counter(
                lookup[r['review_id']]['panel_condition'] for r in rows if all(not x['hit'] for x in r['target']['states']))))
    result['development_outcomes']=misses
    result['held_review_ids']=[d['review_id'] for d in ds if d['decision']=='held_condition_unknown']
    return result


def main():
    dest=OUT/'review.json'
    if dest.exists():
        prior=read(dest);verify(prior);e=read(OUT/'evidence.json');verify(e);validate(e,prior['decisions']);return
    e=read(OUT/'evidence.json');verify(e);ds=decisions(e);validate(e,ds)
    components,paths=component_audit(e)
    frozen(dest,dict(status='condition_review_complete_with_named_gaps',decisions=ds,summary=summarize(e,ds),
        saved_development_component_audit=components,training_component_trace_status='not_completed_per_member_replay_blocked',
        original_training_admission_changed=False,training_started=False,replay_started=False,
        training_ready=False,scope='Targeted condition review, not full-pool or all-label admission. No visible panel is not low contrast or proof of missing asset component.',
        inputs={str(p):file_sha256(p) for p in [OUT/'evidence.json',Path(__file__),*paths]}))
    print(summarize(e,ds))


if __name__=='__main__':main()
