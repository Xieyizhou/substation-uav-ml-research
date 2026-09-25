"""Bind explicit residual observations, validated inference and Chinese conclusion."""
from pathlib import Path
from datetime import datetime,timezone
import statistics,subprocess,sys
from scripts.vision.check_reviewed_hold_fit import OUT,SOURCE,prior,base,setup
from scripts.vision.test_body_material_applicability import baseline_verify

def main():
    setup();p=prior.read(OUT/'protocol.json');s=prior.read(OUT/'summary.json');c=prior.read(OUT/'crosscheck.json');ep=OUT/'bridge-residual-evidence.json';e=prior.read(ep)
    for r in (p,s,c,e):prior.verify(r)
    if c['events'] or len(e['events'])!=1:raise ValueError('Unreviewed additional evidence')
    ev=e['events'][0]
    if ev['event_id']!='B01' or ev['source']['member_id']!='bridge:b9e46c4d1002010d417d5ed144fd75473d919278ae1adc8b943cc6b10db7f093':raise ValueError('Unexpected residual source')
    if prior.file_sha256(ev['page'])!=ev['page_sha256']:raise ValueError('Stale inspected page')
    notes={0:'左侧变压器宽主体、侧面、三个顶部套管及基座可辨，左缘截断；顶部后方杆体不是目标组件。',
           1:'中央电容器组真值对应灰色块体和基座，宽表面清楚，斜阴影跨过表面；后方建筑不归入目标组件。'}
    if len(ev['truth'])!=2 or [t['class_name'] for t in ev['truth']]!=['transformer','capacitor_bank']:raise ValueError('Reviewed full truth changed')
    rp=OUT/'residual-review.json'
    if not rp.exists():
        prior.frozen(rp,dict(status='residual_observations_recorded',evidence_identity=e['identity'],decisions=[dict(event_id='B01',truth=t,
            reason=notes[t['label_line_index']],review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
            status='content_described_not_training_admission',pixel_visibility_certified=False,page_sha256=ev['page_sha256']) for t in ev['truth']],
            inputs={str(ep):prior.file_sha256(ep),ev['page']:ev['page_sha256'],str(Path(__file__).resolve()):prior.file_sha256(__file__)}))
    prior.verify(prior.read(rp))
    paths=[OUT/'protocol.json',OUT/'summary.json',OUT/'crosscheck.json',ep,rp,Path(__file__).resolve(),SOURCE/'evaluation/summary.json']
    prior.verify(prior.read(SOURCE/'evaluation/summary.json'))
    clear={(t['review']['member_id'],t['review']['truth']['annotation_id']) for t in p['targets'] if t['clear_primary'] and t['review']['truth']['class_name']=='reactor'}
    confidence={}
    for key in p['models']:
        path=OUT/'inference'/f'{key}.json';r=prior.read(path);base.valid(r,key,p);paths.append(path);cs=[];ious=[]
        for row in r['rows']:
            sc=row['scoring']
            for m in sc['matches']:
                if (row['member_id'],sc['truth'][m['truth_index']]['annotation_id']) in clear:
                    cs.append(sc['predictions'][m['prediction_index']]['confidence']);ious.append(m['iou'])
        confidence[key]=dict(n=len(cs),mean_confidence=statistics.mean(cs),min_confidence=min(cs),mean_iou=statistics.mean(ious))
    tests=['tests.test_lineage_training_fit','tests.test_structure_fit','tests.test_structure_fit_integrity']
    test=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if test.returncode:raise ValueError(test.stderr)
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline integrity failed')
    paths.append(prior.ROOT/'docs/results/ml_reviewed_hold_fit_diagnosis_20260910.md')
    prior.frozen(OUT/'completion.json',dict(status='diagnosis_complete_condition_coverage_priority',new_training=False,
        model_count=6,pool_members=236,common_exposed_members=s['common_members'],clear_reactor_confidence=confidence,
        baseline=b,regression=dict(output=test.stderr,whole_repository_tested=False),
        conclusion='Strong fitting on reviewed common reactor members with development regression supports condition-limited transfer; not proof of a unique mechanism or unseen-scene failure.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('DIAGNOSIS_COMPLETE_NO_TRAINING',confidence)

if __name__=='__main__':main()
