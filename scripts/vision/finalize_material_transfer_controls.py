"""Summarize six checks and record a bounded, not-ready training design."""
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import subprocess,sys
from scripts.vision.material_transfer_controls import OUT,FIT,PRIOR,prior
from scripts.vision.run_material_scale_control import validate
from scripts.vision.material_control_feasibility import RUN
from scripts.vision.material_member_fit import scoring
from scripts.vision.verify_experiment_baseline import verify as baseline_verify

def scale_order(seed):
    # A balanced multiset of sizes; order cannot depend on observed predictions.
    positions=sorted(range(450),key=lambda i:hashlib.sha256(f'material-multiscale-design-v1:{seed}:{i}'.encode()).hexdigest())
    result=[None]*450
    for rank,pos in enumerate(positions):result[pos]=(320,640,960)[rank%3]
    return result

def main():
    pp=OUT/'protocol.json';p=prior.read(pp);prior.verify(p);ap=OUT/'analytical-controls.json';prior.verify(prior.read(ap))
    bins=defaultdict(Counter);events=[];paths=[pp,ap,Path(__file__).resolve()];source={x['id']:x for x in p['rows']}
    for key in p['models']:
        sp=OUT/'inference'/f'{key}.json';sr=prior.read(sp);validate(sr,p,key);paths.append(sp)
        tp=FIT/'inference'/f'{key}.json';tr=prior.read(tp);prior.verify(tr)
        dp=RUN/'evaluation'/f'{key}.json';dr=prior.read(dp);prior.verify(dr);paths += [tp,dp]
        rows=list(sr['rows'])
        for x in p['rows']:
            if x['role']=='training_fit':old=next(z for z in tr['rows'] if z['member_id']==x['id'])
            else:old=next(z for z in dr['rows'] if z['view_id']==x['view_id'] and z['variant']==x['variant'])
            if old['truth']!=x['truth'] or old['image_sha256']!=x['image_sha256']:raise ValueError('640 reference truth/image mismatch')
            rows.append(dict(id=x['id'],imgsz=640,image_sha256=x['image_sha256'],**scoring(x['truth'],old['predictions'],old['low_predictions'])))
        for row in rows:
            src=source[row['id']];hits={x['truth_index'] for x in row['matches']}
            for i,t in enumerate(row['truth']):
                miss=next((z for z in row['misses'] if z['truth_index']==i),None)
                event=dict(model=key,id=row['id'],role=src['role'],variant=src['variant'],imgsz=row['imgsz'],class_name=t['class_name'],truth_index=i,annotation_id=t['annotation_id'],hit=i in hits,miss=miss)
                events.append(event)
                for cat in ('all',t['class_name']):
                    b=bins[key,src['role'],src['variant'],row['imgsz'],cat];b['truth']+=1;b['hit']+=i in hits
                    if miss:b[miss['reason']]+=1
        # Paired identity includes complete source truth, not cross-world runtime ID.
    index={(x['model'],x['id'],x['annotation_id'],x['imgsz']):x for x in events}
    if len(index)!=len(events):raise ValueError('Duplicate scale target')
    for x in events:
        ref=index[x['model'],x['id'],x['annotation_id'],640]
        x['hit_delta_vs640']=int(x['hit'])-int(ref['hit'])
    planp=RUN/'protocol.json';old=prior.read(planp);prior.verify(old);paths.append(planp)
    design=dict(status='design_only_real_loader_preflight_not_done',training_authorized=False,
        arms=['fixed-640','balanced-320-640-960'],seeds=[7,17,27],optimizer_steps=450,batch=6,
        initialization=old['initialization'],configs={str(s):old['training_config'][f'R-clean-{s}'] for s in (7,17,27)},
        member_sequences={str(s):old['schedules'][f'R-clean-{s}'] for s in (7,17,27)},
        brightness_factors={str(s):old['brightness_factors'][f'R-clean-{s}'] for s in (7,17,27)},
        scale_by_step={str(s):scale_order(s) for s in (7,17,27)},
        single_changed_factor='Input resolution per original batch; members, within-batch order, brightness and counts unchanged',
        qualification='Network sampling/grid and effective image scale change together. Not pure geometric viewpoint augmentation.',
        pixel_budget_ratio_to640=7/6,
        reference_reuse='Only if identities, optimizer config and actual execution environment match. Accelerator change requires both arms re-established; no silent reference reuse.',
        evaluation='Official 640 protocol and full original retention/FPR gates unchanged; 320/960 remain stress diagnostics.',
        required_preflight=['Traverse all 2700 actual members per seed; exact full labels and brightness factors',
            'Assert per-batch height/width equals frozen scale; normalized labels invariant; no hidden second random resize',
            'Recheck held zero, negative positions, per-class and lineage counts',
            'Verify 450 optimizer steps, independent initialization and endpoint-only checkpoint',
            'Validate actual memory/runtime and environment; frozen design is not training readiness'])
    suites=['tests.test_material_transfer_controls','tests.test_material_member_fit','tests.test_exposure_diagnosis','tests.test_material_risk_continuation']
    test=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True,timeout=60)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'completion.json',dict(status='six_controls_complete_next_training_design_not_ready',events=events,
        metrics=[dict(model=k[0],role=k[1],variant=k[2],imgsz=k[3],category=k[4],**v,recall=v['hit']/v['truth']) for k,v in sorted(bins.items())],
        next_design=design,decision='Prioritize controlled multiscale training feasibility/preflight, not a 960 inference switch or more same-scale steps.',
        deferred=['New viewpoint collection: no authorization or need established to start yet','M10 low-exposure capacitor remains a separate exception','No claim scale uniquely explains material/viewpoint transfer'],
        official_evaluation_unchanged=True,training_started=False,selected_candidate=None,tests_output=test.stdout+test.stderr,baseline=b,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(test.stdout+test.stderr);print('SIX_CONTROLS_COMPLETE; NEXT_DESIGN_NOT_TRAINING_READY')

if __name__=='__main__':main()
