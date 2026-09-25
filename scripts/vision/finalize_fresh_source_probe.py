"""Explicit review of four fixed probes; no automatic passes or adaptive expansion."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
from scripts.vision.probe_new_material_sources import OUT,SOURCE,prior
from scripts.vision.evaluate_scale_endpoints import baseline_verify

NOTES={
 'P01-00':('content_observed','绿色电容器主体、顶面和基座完整可见，有杆体投影，不妨碍主体分离。'),
 'P02-00':('content_observed','右后蓝色柜体面板、侧面与顶面可辨，左侧被邻柜遮挡，右图缘截断。'),
 'P02-01':('content_observed','蓝色柜体完整前面板、宽侧面、顶面和基座清楚。'),
 'P02-02':('content_observed','中间蓝色柜体完整背面、顶面和基座清晰。'),
 'P02-03':('content_observed','左图缘柜体宽背面、顶面、基座可见，明显截断。'),
 'P02-04':('pending','右图缘极窄条，主要见地面与基座片段，无法辨认充分柜体主体。'),
 'P03-00':('pending','后排柜体被前景电抗器遮挡，主要余顶部窄带，主体内容不足。'),
 'P03-01':('pending','框中多重柜体与电抗器交叠，目标主要余上部窄带，内容和归属不足。'),
 'P03-02':('pending','后排棕色柜体仅上部面和顶沿，前景柜体占据框的下半；不足以认定充分可辨。'),
 'P03-03':('content_observed','左后柜体较宽主体、顶面和基座部分可辨，右下邻柜遮挡。'),
 'P03-04':('pending','左图缘变压器仅有灰色窄侧片和基座，未见套管，类别内容不足。'),
 'P03-05':('content_observed','前方棕色柜体宽侧面、顶面与基座清楚，右侧电抗器遮挡。'),
 'P03-06':('content_observed','电抗器圆柱主体、顶面及基座完整清晰。'),
 'P04-00':('content_observed','后排棕色柜体主体、面板和基座可辨，左前邻柜部分遮挡。'),
 'P04-01':('content_observed','左侧棕色柜体面板和侧面可辨，前景邻柜遮挡部分主体。'),
 'P04-02':('pending','左图缘仅柜体窄侧片和基座，较大框区域为地面，内容不足。'),
 'P04-03':('content_observed','近处灰色变压器宽主体、顶面、套管和基座清晰。'),
 'P04-04':('content_observed','右侧变压器宽主体、顶面、套管及基座可辨，右边缘截断。'),
 'P04-05':('pending','中后变压器被前景变压器挡住，框内主要为前景顶面及套管，目标只余后方窄片，归属不足。'),
 'P04-06':('content_observed','左中棕色柜体宽侧面及基座可辨，右侧被变压器遮挡。'),
 'P04-07':('pending','电抗器仅余小段灰色曲面及顶面，下部被变压器完全遮挡，内容不足。'),
 'P04-08':('pending','框内大部分是前景灰色变压器，电容器仅疑似左侧绿条和顶沿，不能确认充分主体。'),
}


def main():
    pp=OUT/'protocol.json';cp=OUT/'completion.json';ep=OUT/'review/evidence.json'
    p=prior.read(pp);c=prior.read(cp)
    for r in (p,c,prior.read(ep)):prior.verify(r)
    if {e['review_id'] for f in p['frames'] for e in f['events']}!=set(NOTES):raise ValueError('Review coverage changed')
    now=datetime.now(timezone.utc).isoformat();decisions=[];results=[];paths=[pp,cp,ep,Path(__file__)]
    for f,unit in zip(p['frames'],c['units']):
        sid=f['review_ids'][0]
        if sid!=unit['probe_id']:raise ValueError('Source/result identity mismatch')
        rp=Path(unit['receipt']);r=prior.read(rp);prior.verify(r);paths.append(rp)
        for e in f['events']:
            state,reason=NOTES[e['review_id']];crop=OUT/'review'/(e['review_id']+'.png')
            decisions.append(dict(e,status=state,reason=reason,review_type='AI辅助审核',reviewed_at=now,
                source_image_sha256=prior.file_sha256(f['source_image']),crop_sha256=prior.file_sha256(crop),
                page_sha256=prior.file_sha256(OUT/'review'/(sid+'.png')),whole_image_status='held',training_admitted=False,promotable=False))
        results.append(dict(probe_id=sid,source_member_id=f['member_id'],lineage_id=f['lineage_id'],status='whole_source_held',
            replay_status=r['status'],rgb_exact_all=all(x['rgb_exact'] for x in r.get('records',[])) if r.get('records') else None,
            unboxed_targets=r.get('full_mask_coverage',[{}])[0].get('missing_targets'),
            pending_events=[e['review_id'] for e in f['events'] if NOTES[e['review_id']][0]=='pending']))
    tests=['tests.test_material_mask_coverage','tests.test_material_expansion_gate','tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]+[prior.ROOT/'docs/results/ml_fresh_material_sources_20260911.md']
    old=prior.read(SOURCE/'N05-expansion-v1/reviewed-completion.json');paths.append(SOURCE/'N05-expansion-v1/reviewed-completion.json')
    prior.frozen(OUT/'reviewed-completion.json',dict(status='four_source_probe_complete_no_qualified_source',decisions=decisions,sources=results,
        denied_lineages=sorted(set(old['denied_lineages'])|{f['lineage_id'] for f in p['frames']}),
        training_ready=False,training_started=False,new_material_variants=0,admitted_images=0,
        next_priority='Independently freeze new camera-pose source design with full-scene edge/occlusion screening; confirm actual masks and full boxes before any appearance expansion. Do not silently replace failed probes.',
        baseline=b,regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('PROBES4; LABEL_REVIEWS22; ALL4_HELD; TEST26_PASS; PINNED40_PASS')

if __name__=='__main__':main()
