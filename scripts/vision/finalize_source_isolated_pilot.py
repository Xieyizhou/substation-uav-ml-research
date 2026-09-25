"""Import explicitly inspected pilot observations; do not approve or expand."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from scripts.vision.source_isolated_material_capture import OUT,prior,freeze,verify_capture
from src.ml.artifacts import object_sha256

# Explicit observations from all sixteen full-frame/individual-crop cards viewed in this turn.
NOTES={
 'reactor:original':['灰圆柱主体、顶面及基座完整可辨，无明显前景遮挡。','青色箱体顶侧面和基座可辨，未见正面板。'],
 'reactor:warm':['暖色圆柱完整可辨，明暗侧面边界清楚。','暖色箱体侧背面可辨，无面板特征。'],
 'reactor:cool':['冷灰圆柱顶面、侧面及基座清楚。','冷灰箱体完整侧背面，与圆柱分离。'],
 'reactor:neutral':['中性灰圆柱轮廓和底座可辨，背景墙未遮主体。','中性灰箱体侧背面清楚，未见面板。'],
 'capacitor_bank:original':['左后柜体及深色面板可辨。','右后柜体侧背面清楚，左下部邻接前景箱体。','计划目标大箱体和底座清楚，未见独立电容器圆柱。'],
 'capacitor_bank:warm':['左后暖色柜体面板边界较弱但可见。','右后暖色箱体可辨，未见面板。','暖色计划箱体完整，未见外露电容器组件。'],
 'capacitor_bank:cool':['左后冷色柜体面板轮廓可见。','右后冷灰柜体侧背面可辨。','冷灰计划箱体大面积清楚，未见外露圆柱结构。'],
 'capacitor_bank:neutral':['左后灰柜面板边缘可见但对比弱。','右后灰箱体可辨，与前景边界相邻。','灰色计划箱体完整，外露组件未见。'],
 'switchgear:original':['青色大箱体、顶面与基座完整可辨；当前视角不呈现正面板。'],
 'switchgear:warm':['暖色箱体主体无遮挡，侧背面无可辨面板细节。'],
 'switchgear:cool':['冷色箱体完整且与地面分离，无正面板证据。'],
 'switchgear:neutral':['灰色箱体顶侧面和底座可辨，未见面板。'],
 'transformer:original':['俯视主体、三根顶部套管和底座清楚；不是仅基座样本。'],
 'transformer:warm':['暖色主体与三根浅色套管清楚，未见明显遮挡。'],
 'transformer:cool':['冷色主体及三个顶部套管可辨，完整框与图缘分离。'],
 'transformer:neutral':['灰色主体、顶面三根套管及底座可辨，侧面特征较少。'],
}
EXPECTED={'reactor':['reactor_north','entry_switchgear'],
          'capacitor_bank':['west_switchgear_04','west_switchgear_03','capacitor_east'],
          'switchgear':['west_switchgear_02'],'transformer':['transformer_sw']}


def containment(world):
    m=ET.parse(world).find('./world/model[@name="capacitor_east"]/link')
    body=m.find('visual[@name="body"]');pos=list(map(float,body.findtext('pose').split()))
    size=list(map(float,body.findtext('geometry/box/size').split()))
    if any(pos[3:]):raise ValueError('unsupported_rotated_body')
    bb=[value for i in range(3) for value in (pos[i]-size[i]/2,pos[i]+size[i]/2)]
    rows=[]
    for v in m.findall('visual'):
        if not v.get('name','').startswith('capacitor_'):continue
        p=list(map(float,v.findtext('pose').split()))
        if any(p[3:]):raise ValueError('unsupported_rotated_component')
        radius=float(v.findtext('geometry/cylinder/radius'));length=float(v.findtext('geometry/cylinder/length'))
        b=[p[0]-radius,p[0]+radius,p[1]-radius,p[1]+radius,p[2]-length/2,p[2]+length/2]
        rows.append(dict(component=v.get('name'),bounds=b,inside_body=all(bb[i]<b[i] and b[i+1]<bb[i+1] for i in (0,2,4))))
    if len(rows)!=6:raise ValueError('unexpected_component_count')
    return dict(body_bounds=bb,components=rows,all_six_inside=all(r['inside_body'] for r in rows),
        interpretation='Closed-body asset convention, not proof of erroneous labels. Cannot supply exterior cylinder evidence without a separately authorized asset change.')


def validate_decisions(evidence, decisions):
    expected={x['event_id']:x for e in evidence for x in e['events']}
    if len(decisions)!=len(expected) or len({d['event_id'] for d in decisions})!=len(decisions):raise ValueError('missing_or_duplicate_review')
    for d in decisions:
        e=expected[d['event_id']]
        if d['crop_sha256']!=e['crop_sha256'] or d['truth_identity']!=object_sha256(e['truth']):raise ValueError('stale_review')
        if not d['reason'] or d['training_admitted']:raise ValueError('invalid_review')


def main():
    p=freeze();capture=prior.read(OUT/'pilot-completion.json');prior.verify(capture)
    ec=prior.read(OUT/'evidence/completion.json');prior.verify(ec)
    if len(capture['units'])!=16 or any(not c['passed'] for c in ec['pair_checks']):raise ValueError('incomplete_or_misaligned_pilot')
    evidence=[];decisions=[];frames=[];paths=[OUT/'protocol.json',OUT/'pilot-completion.json',OUT/'evidence/completion.json',Path(__file__)]
    for u in (u for u in p['units'] if u['pilot']):
        verify_capture(u,OUT/'captures'/u['key']/'collection-receipt.json')
        ep=OUT/'evidence'/u['key']/'evidence.json';e=prior.read(ep);prior.verify(e);evidence.append(e);paths.append(ep)
        category=u['pair_id'].split(':')[1];notes=NOTES[category+':'+u['variant']]
        if [x['object_id'] for x in e['events']]!=EXPECTED[category]:raise ValueError('review_instance_order_changed')
        if len(notes)!=len(e['events']):raise ValueError('review_inventory_changed')
        for x,note in zip(e['events'],notes):
            decisions.append(dict(event_id=x['event_id'],object_id=x['object_id'],reason=note,
                status='visible_body_reviewed_not_admitted',review_nature='AI辅助审核',
                reviewed_at=datetime.now(timezone.utc).isoformat(),crop_sha256=x['crop_sha256'],
                truth_identity=object_sha256(x['truth']),evidence_sha256=prior.file_sha256(ep),training_admitted=False))
        frames.append(dict(key=u['key'],full_frame_visually_inspected=True,
            unboxed_targets='not_visually_confirmed',pixel_visibility_certified=False,
            content_gap='exterior_capacitor_components_absent' if category=='capacitor_bank' else 'front_panel_not_shown' if category=='switchgear' else None))
    validate_decisions(evidence,decisions)
    maskp=OUT/'edge-replay-v1/replay/edge-pilot/attempt-01/receipt.json';mask=prior.read(maskp);prior.verify(mask);paths.append(maskp)
    if mask['status']!='original_pixel_evidence_certified' or any(x['missing_targets'] for x in mask['full_mask_coverage']):raise ValueError('edge_replay_unresolved')
    frames[0]['pixel_visibility_certified']=True
    geometry=containment(OUT/'plans/layout-A/original/world.sdf')
    if not geometry['all_six_inside']:raise ValueError('unsupported_geometry_conclusion')
    from scripts.vision.evaluate_same_source_material_dose import baseline_verify
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('baseline_changed')
    tests=['test_source_isolated_material_capture','test_source_isolated_pilot_review','test_canonical_gates','test_canonical_recovery','test_canonical_shutdown','test_full_scene_pose_design']
    result=subprocess.run([sys.executable,'-m','unittest',*['tests.'+t for t in tests]],capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    paths.extend(prior.ROOT/'tests'/f'{t}.py' for t in tests)
    report=prior.ROOT/'docs/results/ml_source_isolated_material_pilot_20260913.md';paths.append(report)
    dest=OUT/'reviewed-pilot-completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='pilot_captured_reviewed_expansion_held_for_structure_coverage',captured_images=16,
        planned_images=64,uncaptured_images=48,decisions=decisions,frames=frames,geometry_check=geometry,
        pair_checks=ec['pair_checks'],baseline=baseline,tests=dict(exit_code=result.returncode,output=result.stdout+result.stderr),
        expansion_allowed=False,training_ready=False,training_started=False,
        limitations=['Pilot is four poses from layout A only; layout B frozen but not rendered.',
            'Shared primitive assets; no independent new asset coverage.',
            'Exact mask evidence applies only to reactor original frame, not its variants.',
            'Full-image development/pool pixel dedup and training quota feasibility are not completed; no admission.'],
        next_action='Clarify closed-cabinet exterior coverage versus new visible-component assets before expansion; do not modify existing assets or labels.',
        inputs={str(path):prior.file_sha256(path) for path in paths}))
    print('PILOT_16_CAPTURED_28_LABELS_REVIEWED_EXPANSION_HELD')


if __name__=='__main__':main()
