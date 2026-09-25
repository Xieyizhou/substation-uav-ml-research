"""Explicit FP correspondence, descriptive exposure and paired fit diagnostics."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
from scripts.vision.structure_fit import OUT,SOURCE,read,verify,frozen,file_sha256,unique,truth_for
from scripts.vision.check_structure_fit_sources import strict_review,strict_unit
from src.ml.artifacts import object_sha256

# Manually selected correspondences to the already inspected ROI contents.
# No automatic 'coverage sufficient' decisions or simulator instance certificates.
LINKS = {
 1:('mixed_boundary',[15,28,60],'天空占主导、围墙与右杆；训练侧仅墙地影组合近似，比例和图缘条件未确认。'),
 2:('mixed_boundary',[15,28,60],'天空、墙地与左缘窄杆；尚无已逐ROI确认的同等细边条件。'),
 3:('mixed_boundary',[15,28,60],'与 F01 同图结构，重复预测不增加独立样本。'),
 4:('mixed_boundary',[15,28,60],'天空围墙和右杆组合，训练只确认组成元素而非该组合覆盖充分。'),
 5:('mixed_boundary',[15,28,60],'低光天空围墙杆体；两光照训练变体有近似边界，确切组合待定。'),
 6:('mixed_boundary',[15,28,60],'低光天空围墙杆体；不将整帧主题当框内对象。'),
 7:('mixed_boundary',[15,28,60],'斜墙、天空和地面组合，只有部分条件对应。'),
 8:('blue_side',[5,49,58],'左缘蓝柜背侧截片；训练有蓝柜侧背及图缘截断，但左右边和片段尺度不相同。'),
 9:('mixed_boundary',[21,42],'天空、杆、墙和灰块局部混合；训练有遮挡重叠，但预测覆盖的天空比例不同。'),
 10:('blue_front',[6,19,59],'蓝柜深色正面面板，训练确认蓝柜面板及多尺度视角。'),
 11:('blue_front',[6,19,59],'远蓝柜面板；远景 N117/N118 提供具名证据，不证明尺寸覆盖充分。'),
 12:('pole_crossbar',[13,46,54],'两竖杆及顶部横件；训练有细杆横件，组合几何未证明等同。'),
 13:('blue_side',[5,6,49],'蓝柜宽侧、窄面板、顶部和基座；侧面与面板都有训练证据。'),
 14:('mixed_boundary',[15,28,60],'天空、斜杆和围墙组合，训练近似而非完整条件对应。'),
 15:('gray_base',[14],'只覆盖灰块下侧和基座，N027/N028 为对应局部结构。'),
 16:('gray_front_side',[30,40,45],'灰块深面板和宽侧面；属于 control_building 外形，不改称 cabinet 资产。'),
 17:('gray_front_side',[30,42,45],'灰块宽侧和右面板，训练有相同资产的近似视角。'),
 18:('gray_front_side',[30,42,45],'低光灰块宽侧和右面板；两种训练光照均保留。'),
 19:('gray_front',[27,30,40],'灰块正面深面板；存在训练对应，不能宣布结构缺失。'),
 20:('gray_front',[27,30,40],'低光灰块正面深面板；存在两光照训练对应。'),
 21:('blue_front',[6,19,59],'低光蓝柜前面板；与灰建筑面板分开记资产身份。'),
 22:('gray_front_side',[31,42,45],'灰块宽侧及右侧面板，训练存在类似主体及面板。'),
 23:('gray_occluded',[21,42],'灰块背侧与前景杆遮挡，训练 N041/N042、N083/N084 有具名遮挡证据。'),
 24:('blue_front',[6,19,59],'同一蓝面板跨 seed 重复，沿用该图本轮已绑定逐框观察。'),
 25:('blue_front',[6,19,59],'同一远蓝面板跨 seed 重复，独立图数不增加。'),
 26:('gray_front_side',[30,42,45],'灰块宽侧及右面板，训练存在近似结构。'),
 27:('gray_front_side',[30,42,45],'低光灰块宽侧和面板；预测类别变压器不改变源资产身份。'),
 28:('mixed_boundary',[15,28,60],'低光斜墙天空地面混合；训练仅部分条件对应。'),
 29:('gray_front',[27,30,40],'灰块正面面板有具名训练证据。'),
 30:('gray_front',[27,30,40],'低光灰块正面面板有具名训练证据。'),
 31:('gray_edge',[3,17,24,31],'右缘灰面板及基座截片；训练已有右缘近景片段，但片段比例仍应区别。'),
 32:('gray_base_mixed',[14,32],'灰块基座边和地面混合；不能将整框记成完整柜体。'),
 33:('edge_pole',[],'右缘极窄杆体；全图中有图缘杆的描述，但尚无单独 ROI 足以认证该细片段覆盖。'),
 34:('edge_pole',[],'同图极窄杆重复框；待定结构不因重复 seed 或框数获得覆盖。'),
 35:('edge_pole',[],'同图同结构重复；需补充细边 ROI 证据，不据此宣布训练中不存在。'),
 36:('gray_edge',[20,32,36],'右缘灰块背侧被裁切，训练已确认该类图缘条件。'),
 37:('gray_front_side',[30,42,45],'灰块宽侧和右面板，训练有对应。'),
 38:('blue_front',[6,19,59],'与 F10/F24 同一蓝柜面板跨 seed 重复。'),
 39:('blue_side',[5,6,49],'蓝柜宽侧及小面板，不把宽侧覆盖当作面板充分覆盖。'),
 40:('gray_front_side',[30,42,45],'灰块宽侧和右面板，训练有近似视角。'),
 41:('gray_front_side',[30,42,45],'低光灰块宽侧和右面板，训练两光照对应。'),
 42:('gray_front',[27,30,40],'灰块面板局部、不是完整顶面；对应面板 ROI。'),
}


def main():
    p=read(OUT/'protocol.json');ev=read(OUT/'evidence.json');review=read(OUT/'review.json')
    for doc in (p,ev,review):verify(doc)
    strict_review(ev,review['decisions']);ds=unique(review['decisions'],lambda d:d['event_id'])
    np=SOURCE/'post-training-review-v1/manifest.json';op=SOURCE/'post-training-review-v1/review.json'
    old=read(np);od=read(op);verify(old);verify(od)
    olddec=unique(od['decisions'],lambda d:d['event_id'])
    ns=unique(p['negative'],lambda n:n['event_id'])
    inputs={str(x):file_sha256(x) for x in (OUT/'protocol.json',OUT/'evidence.json',OUT/'review.json',OUT/'source-audit.json',OUT/'member-source-trace.json',np,op,Path(__file__))}
    rows=[]
    if len(LINKS)!=42 or len(old['negative'])!=42:raise ValueError('FP scope differs')
    for e in old['negative']:
        d=olddec[e['event_id']]
        for key in ('image_sha256','evidence_sha256'):
            if d[key]!=e[key]:raise ValueError('Stale FP decision')
        for kind in ('image','evidence'):
            if file_sha256(e[kind+'_path'])!=e[kind+'_sha256']:raise ValueError('Stale FP evidence')
            inputs[e[kind+'_path']]=e[kind+'_sha256']
        if d['source_event_identity']!=object_sha256(e):raise ValueError('Stale FP prediction')
        family,pairs,reason=LINKS[int(e['event_id'][1:])]
        training=[]
        for pair in pairs:
            for number in (2*pair-1,2*pair):
                nid=f'N{number:03}';n=ns[nid];decision=ds[nid]
                training.append(dict(event_id=nid,member_id=n['member']['member_id'],image_sha256=n['member']['image_sha256'],
                    pair_id=n['source']['pair_id'],source_map=n['source']['map_id'],roi_evidence={s:v for s,v in decision['structures'].items() if v['state']=='present'},
                    exposures={k:Counter(m['draws'])[n['member']['member_id']] for k,m in p['models'].items()}))
        rows.append(dict(event=e,previous_decision=d,review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            visual_structure=family,simulator_asset='control_building' if family.startswith('gray') else 'cabinet_center' if family.startswith('blue') else 'unknown_or_multiple',
            attribution_limit='saved-world and visual correspondence; no instance-mask certification',
            reason=reason,training_counterparts=training,coverage_status='named_correspondence_not_sufficiency' if training else 'unresolved_fine_ROI_gap_not_confirmed_absence'))
    frozen(OUT/'fp-structure-map.json',dict(status='all_42_FP_have_named_correspondence_or_gap',events=rows,
        unique_development_images=len({r['event']['image_sha256'] for r in rows}),unique_development_views=len({r['event']['view_id'] for r in rows}),
        confirmed_absence_claims=0,unknown_FP_ROI_events=[r['event']['event_id'] for r in rows if not r['training_counterparts']],inputs=inputs))

    # Paired member and annotation identity, not array-index correspondence.
    models={}
    for key in p['models']:
        path=OUT/'inference'/f'{key}.json';m=read(path);strict_unit(m,key,p);models[key]=m;inputs[str(path)]=file_sha256(path)
    common=set.intersection(*[{r['member_id'] for r in m['pool'] if r['actual_exposures']} for k,m in models.items() if k!='v2.11'])
    membermap={r['member_id']:r for r in p['rows']};matrix=[];paired=[];strata={}
    for key,m in models.items():
        items=[];counts=Counter(p['models'][key]['draws'])
        for r in m['pool']:
            mr=membermap[r['member_id']]
            with Image.open(mr['image_path']) as im:w,h=im.size
            hits={r['truth'][a['truth_index']]['annotation_id'] for a in r['matches']}
            for t in r['truth']:
                b=t['bbox_xyxy'];short=min(b[2]-b[0],b[3]-b[1])*640/max(w,h)
                items.append(dict(member_id=r['member_id'],annotation_id=t['annotation_id'],class_name=t['class_name'],short_side_640=short,
                    size_bucket='lt32' if short<32 else '32to64' if short<64 else 'ge64',actual_exposures=counts[r['member_id']],
                    seen=bool(counts[r['member_id']]),hit=t['annotation_id'] in hits,registered_lineage=mr['lineage_id']))
        windows=[]
        for start in range(0,len(p['models'][key]['draws']),300):
            c=Counter(p['models'][key]['draws'][start:start+300]);bins=Counter()
            for i in items:bins[(i['class_name'],i['size_bucket'])]+=c[i['member_id']]
            windows.append(dict(step_start=start//6+1,step_end=(start+300)//6,class_size_instance_exposures=[dict(class_name=k[0],size_bucket=k[1],exposures=v) for k,v in sorted(bins.items())]))
        strata[key]=dict(instances=items,windows=windows)
        if key=='v2.11':continue
        references=['v2.11',f"retained_reference-450-{key.rsplit('-',1)[1]}"]
        if key.startswith('interleaved'):references.append(key.replace('interleaved','staged'))
        current={i['annotation_id']:i for i in items}
        for ref in references:
            if ref==key:continue
            rm={r['member_id']:r for r in models[ref]['pool']}
            changes=[]
            for row in m['pool']:
                if row['member_id'] not in common:continue
                a=rm[row['member_id']];oldhits={a['truth'][t['truth_index']]['annotation_id'] for t in a['matches']}
                for t in row['truth']:
                    before=t['annotation_id'] in oldhits;after=current[t['annotation_id']]['hit']
                    changes.append(dict(member_id=row['member_id'],annotation_id=t['annotation_id'],class_name=t['class_name'],
                        result='retained' if before and after else 'gain' if after else 'loss' if before else 'persistent_miss'))
            paired.append(dict(model=key,reference=ref,common_unique_members=len(common),events=changes,counts=dict(Counter(c['result'] for c in changes))))
    inputs[str(OUT/'fp-structure-map.json')]=file_sha256(OUT/'fp-structure-map.json')
    frozen(OUT/'diagnostic-matrix.json',dict(status='all_output_and_actual_exposure_reverified',instance_size_and_exposure=strata,common_member_pairing=paired,
        formal_matching_competition_events=[dict(model=k,cohort=co,frame=r.get('member_id',r.get('view_id')),variant=r.get('variant'),miss=mi) for k,m in models.items() for co in ('pool','development') for r in m[co] for mi in r['misses'] if mi['formal_matching_competition']],inputs=inputs))
    print('STRUCTURE_MAP_AND_MATRIX_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    if args.summarize:main()
    else:print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
