"""Authored observations of 12 viewed pages and 19 crops; never infer approval from metadata."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.prepare_clear_context_increment import OUT,checked

BOXES={
 'C01':['变压器大块主体、三个顶端柱及基座完整，主体无明显前景遮挡。'],
 'C02':['变压器主体、顶端三柱、基座清楚。','电容器组的顶面及两侧主体可辨，右侧被杆体局部遮挡，非仅细片或基座。'],
 'C03':['近处变压器顶面、三个顶柱、正侧主体和基座清楚。','右后电容器主体和顶面可辨，左下被近变压器遮挡，底座右段可见。'],
 'C04':['开关柜背/侧面、顶面及基座清楚；未见前面板，类别依据唯一来源身份，不能宣称具有独特判别外形。'],
 'C05':['开关柜前面板、箱体和基座完整。','后方电抗器圆柱、顶面和基座可辨，左下被开关柜遮挡。'],
 'C06':['开关柜顶面、侧/背面及基座完整；无前面板证据，保留背侧视角限制。'],
 'C07':['电容器块状主体、窄顶面和基座完整，无明显目标前景遮挡；简单几何不保证类别外形唯一。'],
 'C08':['电容器块状主体、顶面和基座完整；类别由来源实例确认，不仅凭颜色。'],
 'C09':['电容器两个侧面、顶面及基座清楚，非图缘片段。'],
 'C10':['俯视开关柜顶面、侧面和基座清楚，前面板未见。','电抗器圆形顶面、圆柱侧面和基座完整。'],
 'C11':['变压器两侧主体、顶面、三柱与基座完整。','右侧电抗器圆柱、顶面和基座清楚。','后方电容器顶面与上部两侧主体可辨，下部被变压器遮挡，非完整无遮挡实例。'],
 'C12':['开关柜顶面、前面板和主体基座可辨。','电抗器顶面、圆柱和基座完整。'],
}
FULL={
 'C01':'全图只有已框变压器；其他可见部分为围墙、地面网格和阴影。',
 'C02':'两处目标均有框。左侧未框青色块体在来源世界为cabinet_west普通柜体，位置投影吻合；不按外形将其改标为开关柜。右前杆体非目标。',
 'C03':'两处目标均有框。左后灰色块体对应control_building，前后杆体及围墙不是四类目标。',
 'C04':'已框开关柜外可见围墙、地面和边界线，没有发现其他独立目标内容。',
 'C05':'开关柜、电抗器均有框。右缘青色箱体对应cabinet_north普通柜体；不把普通柜体归入目标。',
 'C06':'已框开关柜外为围墙、地面和绿色标记，未发现未框目标。',
 'C07':'唯一目标已框，其余为杆体、围墙、地面与阴影。',
 'C08':'唯一目标已框，其余为围墙、地面与阴影。',
 'C09':'唯一目标已框。左后灰色面板块体对应control_building；其他是杆体、围墙和地面。',
 'C10':'两处目标已框。下缘青色截断块体对应cabinet_north普通柜体；其截断不属于遗漏的四类目标标签。',
 'C11':'三处目标有框。左缘灰色片段对应control_building；其余为杆体、围墙、地面及阴影。',
 'C12':'两处目标有框。右侧外形近似的青色面板柜为cabinet_north；实例身份可追溯，但这种相近资产外形会限制结构区分，不认为数据已解决这一问题。',
}


def frozen_evidence():
    dest=OUT/'evidence/manifest-v2.json'
    if dest.exists():return checked(dest)
    old=checked(OUT/'evidence/manifest.json');runtime=checked(OUT/'capture-run-003.json')
    binary=Path('/tmp/substation-uav-gz-rgbd-bridge');snapshot=OUT/'native-bridge-attempt-003'
    if file_sha256(binary)!=runtime['bridge_binary_sha256']:raise ValueError('Runtime binary changed before snapshot')
    if not snapshot.exists():shutil.copy2(binary,snapshot)
    if file_sha256(snapshot)!=runtime['bridge_binary_sha256']:raise ValueError('Snapshot mismatch')
    rp=OUT/'runtime-snapshot.json'
    if not rp.exists():
        paths=[snapshot,OUT/'toolchain-attempt-003.json',OUT/'capture-attempt-003/collection-receipt.json',Path(__file__)]
        write_record(rp,dict(status='immutable_runtime_snapshot_verified',supersedes_runtime_reference='capture-run-003.json',
            erratum='Original run metadata bound a mutable /tmp compiler output. This independent receipt preserves the observed binary as an immutable experiment-local snapshot; original collection and logs unchanged.',
            observed_runtime_identity=runtime['identity'],binary_sha256=runtime['bridge_binary_sha256'],
            training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in paths}))
    record={k:v for k,v in old.items() if k not in ('identity','inputs')}
    deps={path:h for path,h in old['inputs'].items() if not Path(path).name.startswith('capture-run-')}
    deps[str(rp.resolve())]=file_sha256(rp)
    record.update(inputs=deps,supersedes_manifest_identity=old['identity'],erratum='Runtime provenance uses immutable snapshot, not mutable /tmp artifact. Visual evidence unchanged.')
    return write_record(dest,record)


def validate(e,r):
    frames={f['review_id']:f for f in e['frames']};decisions=r['frames']
    if len(decisions)!=len(frames) or {d['review_id'] for d in decisions}!=set(frames):raise ValueError('Missing/duplicate frame review')
    for d in decisions:
        f=frames[d['review_id']]
        for key in ('image_sha256','truth_sha256','page_sha256'):
            if d[key]!=f[key]:raise ValueError('Stale review evidence')
        if d['decision']!='accepted_for_bounded_training_candidate' or not d['full_frame_reason']:raise ValueError('Unresolved frame')
        if len(d['boxes'])!=len(f['objects']):raise ValueError('Missing label review')
        for a,b in zip(d['boxes'],f['objects']):
            if any(a[k]!=b[k] for k in ('annotation_id','scene_device_id','bbox_xyxy','crop_sha256')):raise ValueError('Box identity drift')
            if a['state']!='visible_identifiable_content' or not a['reason']:raise ValueError('Unresolved content')


def run():
    e=frozen_evidence();dest=OUT/'review.json'
    if dest.exists():r=checked(dest);validate(e,r);return r
    if set(BOXES)!=set(FULL) or set(BOXES)!={f['review_id'] for f in e['frames']}:raise ValueError('Authored review incomplete')
    rows=[]
    for f in e['frames']:
        reasons=BOXES[f['review_id']]
        if len(reasons)!=len(f['objects']):raise ValueError('Authored box count mismatch')
        rows.append(dict(review_id=f['review_id'],view_id=f['view_id'],decision='accepted_for_bounded_training_candidate',
            image_sha256=f['image_sha256'],truth_sha256=f['truth_sha256'],page_sha256=f['page_sha256'],
            full_frame_reason=FULL[f['review_id']],full_frame_viewed=True,
            boxes=[dict(**o,state='visible_identifiable_content',reason=reason,pixel_visibility_certified=False) for o,reason in zip(f['objects'],reasons)]))
    paths=[OUT/'evidence/manifest-v2.json',OUT/'plan/plan.json',Path(__file__)]
    result=dict(status='twelve_frames_nineteen_labels_reviewed_for_bounded_candidate_export',frames=rows,
        review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
        limits=['No pixel-level mask certification or claim of complete absence of occlusion.','Cabinet-like target and non-target assets can be visually similar; source identity does not establish unique visual separability.','12 new camera poses share an existing development layout/assets. Not 12 independent scenes.'],
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in paths})
    validate(e,result);return write_record(dest,result)


if __name__=='__main__':print(run()['status'])
