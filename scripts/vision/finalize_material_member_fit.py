"""Summarize observed fit and existing development predictions without new gates."""
from collections import Counter,defaultdict
from pathlib import Path
import subprocess,sys
from scripts.vision.material_member_fit import OUT,PRIOR,preflight,prior
from scripts.vision.infer_material_member_fit import validate_unit
from scripts.vision.review_material_member_fit import validate
from scripts.vision.verify_experiment_baseline import verify as baseline_verify

def main():
    p=preflight();rp=OUT/'review.json';ds=validate(p,prior.read(rp));paths=[OUT/'protocol.json',rp,Path(__file__).resolve()]
    events={(e['member_id'],e['truth']['annotation_id']):e for e in p['events']};cells={};detail=[]
    for key in p['models']:
        path=OUT/'inference'/f'{key}.json';r=prior.read(path);validate_unit(r,key,p);paths.append(path);groups=defaultdict(Counter)
        for x in r['rows']:
            m=next(m for m in p['members'] if m['member_id']==x['member_id']);hits={a['truth_index'] for a in x['matches']}
            for i,t in enumerate(x['truth']):
                e=events[x['member_id'],t['annotation_id']];decision=ds[e['event_id']]
                miss=next((v for v in x['misses'] if v['truth_index']==i),None)
                bbox=t['bbox_xyxy']
                detail.append(dict(model=key,event_id=e['event_id'],member_id=x['member_id'],lineage_id=m['lineage_id'],object_id=e['object_id'],
                    source_variant=m['source_variant'],class_name=t['class_name'],content=decision['content'],hit=i in hits,miss=miss,
                    actual_exposures=x['actual_exposures'],role='initialization_comparison' if key=='v2.11' else ('training_fit' if x['actual_exposures'] else 'unexposed_comparison'),
                    bbox_xyxy=bbox,primary_gray=m['primary_gray']))
                for cat in ('all',t['class_name']):
                    for group in ('all_selected',m['source_variant']):
                        g=groups[group,cat];g['truth']+=1;g['hit']+=i in hits
                        if miss:g[miss['reason']]+=1
        cells[key]=[dict(group=k[0],category=k[1],**v,recall=v['hit']/v['truth']) for k,v in groups.items()]
    devp=PRIOR/'recomputed-metrics.json';dev=prior.read(devp);prior.verify(dev);paths.append(devp)
    suites=['tests.test_material_member_fit','tests.test_material_risk_continuation','tests.test_material_control_feasibility','tests.test_exposure_diagnosis','tests.test_pixel_duplicates']
    test=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True,timeout=60)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity failure')
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    for name in ('infer_material_member_fit.py','review_material_member_fit.py'):paths.append(prior.ROOT/'scripts/vision'/name)
    prior.frozen(OUT/'completion.json',dict(status='bounded_training_fit_diagnosis_complete',cells=cells,instance_results=detail,
        development_cells=dev['cells'],development_events=dev['events'],
        independent_pose_groups=4,training_images=12,training_truths=42,weights=7,
        fit_is_not_generalization=True,selected_candidate=None,training_started=False,
        next_priority='Design a source-grouped material/viewpoint diversity control; do not add steps to the same four poses as the default fix.',
        exception='M10 capacitor is missed by all six trained models despite broad discernible body; each model saw this image once. This does not establish a sufficient exposure threshold.',
        limits=['Same assets/layout; no novel-scene inference','Training fit assessed on native unaugmented images, not every historical brightness tensor',
            '42 explicit visual content reviews do not certify per-pixel masks or authorize training intake',
            'Initialization counts=0 only denotes no exposure in these current runs, not proof about all prior baseline training'],
        tests_output=test.stdout+test.stderr,baseline=baseline,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(test.stdout+test.stderr);print('DIAGNOSIS_COMPLETE; PINNED40_PASS; NO_TRAINING')

if __name__=='__main__':main()
