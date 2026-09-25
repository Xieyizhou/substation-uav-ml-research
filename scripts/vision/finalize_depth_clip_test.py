"""Frozen four-pose technical acceptance, separate from production/data admission."""
import subprocess
import sys
from pathlib import Path
import psutil
from scripts.vision.run_depth_clip_test import OUT,BUILD,controls,base
from scripts.vision.test_body_material_applicability import baseline_verify


def compare(tag,original,depth,six_plane):
    if depth['status']!='existing_pose_technical_checks_passed' or not depth['process_cleanup_complete'] or len(depth['records'])!=3:
        raise ValueError('Depth replay incomplete')
    if len(original['records'])!=3 or len(six_plane['records'])!=3:raise ValueError('Reference incomplete')
    rows=[]
    for a,b,c in zip(original['records'],depth['records'],six_plane['records']):
        if not all(x['rgb_exact'] and x['skew_ms']<=33.334001 for x in (a,b,c)):raise ValueError('RGB/time comparison invalid')
        old,new,full=a['full_boxes'],b['full_boxes'],c['full_boxes']
        expected=set(old)|({'128'} if tag=='T027' else set())
        deltas={k:max(abs(x-y) for x,y in zip(old[k],new[k])) for k in set(old)&set(new)}
        six_deltas={k:max(abs(x-y) for x,y in zip(old[k],full[k])) for k in set(old)&set(full)}
        rows.append(dict(exact_expected_membership=set(new)==expected,
            lost=sorted(set(old)-set(new)),added=sorted(set(new)-set(old)),
            original_to_depth_deltas=deltas,original_to_six_plane_deltas=six_deltas,
            max_depth_delta=max(deltas.values(),default=0),max_six_plane_delta=max(six_deltas.values(),default=0),
            recovered_128=new.get('128') if tag=='T027' else None))
    return dict(review_id=tag,technical_gate_passed=all(x['exact_expected_membership'] and x['max_depth_delta']<=1 for x in rows),comparisons=rows)


def main():
    base.guard()
    paths=[OUT/'protocol.json',OUT/'fenced-run-receipt.json',OUT/'clock-fence-protocol.json',OUT/'run-receipt.json']
    for p in paths:base.verify(base.read(p))
    summary=base.read(OUT/'fenced-run-receipt.json')
    if summary['status']!='runs_complete' or len(summary['results'])!=4:raise ValueError('Four complete runs required')
    results=[]
    for unit in summary['results']:
        rp=Path(unit['receipt']);base.verify(base.read(rp));depth=base.read(rp)
        tag=unit['review_id'];cp=controls(tag);sp=controls(tag,'clipped')
        for p in (cp,sp):base.verify(base.read(p))
        original,six=base.read(cp),base.read(sp)
        result=compare(tag,original,depth,six)
        for old,new,other in zip(original['records'],depth['records'],six['records']):
            masks=[folder/f'frame-{row["capture_index"]}-mask.bin' for folder,row in ((cp.parent,old),(rp.parent,new),(sp.parent,other))]
            if len({base.file_sha256(p) for p in masks})!=1:raise ValueError('Instance masks differ across builds')
        paths += [rp,cp,sp,rp.parent/'clock-preflight.json']
        base.verify(base.read(paths[-1]))
        for p in depth['loaded_libraries']:
            path=Path(p)
            if not path.is_file():raise ValueError('Loaded library no longer available')
            paths.append(path)
        results.append(result)
    if [r['review_id'] for r in results]!=['T020','T036','T023','T027']:raise ValueError('Unexpected pose order')
    cpp=subprocess.run([str(OUT/'test-depth-clip')],capture_output=True,text=True,timeout=10)
    if cpp.returncode:raise ValueError(cpp.stderr)
    tests=['tests.test_depth_clip_acceptance','tests.test_edge_raster_validation','tests.test_near_clip_build','tests.test_full_box_runtime_trace','tests.test_full_box_projection','tests.test_dual_box_diagnosis','tests.test_edge_box_trace','tests.test_instance_visibility_diagnosis']
    run=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=base.ROOT,capture_output=True,text=True,timeout=60)
    if run.returncode:raise ValueError(run.stderr)
    baseline=baseline_verify(base.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    if any(p.info['exe']==str(OUT/'diagnostic-server') for p in psutil.process_iter(['exe'])):raise ValueError('Server still running')
    paths += [Path(__file__),base.ROOT/'docs/results/ml_depth_clip_test_20260910.md']
    paths += [base.ROOT/(t.replace('.','/')+'.py') for t in tests]
    dest=OUT/'completion.json'
    if dest.exists():base.verify(base.read(dest));print('VALID_DEPTH_ONLY_COMPLETION_REUSED');return
    passed=all(r['technical_gate_passed'] for r in results)
    base.frozen(dest,dict(status='four_pose_depth_clip_technical_gate_passed_not_production_ready' if passed else 'four_pose_depth_clip_regression_held',
        pose_comparisons=results,technical_gate_passed=passed,
        cpp_checks=dict(count=8,returncode=cpp.returncode,stdout=cpp.stdout),
        regression=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,whole_repository_tested=False),baseline=baseline,
        production_ready=False,training_ready=False,training_started=False,historical_labels_changed=False,production_library_written=False,
        interpretation='Depth-only clipping recovers the known missing instance while retaining original full_2d coordinate semantics on the four frozen poses. This is a renderer technical result, not model accuracy or training admission.',
        outstanding=['Extreme truncation supervision and independent annotation revision need explicit review.',
            'The known one-pixel instance boundary ambiguity remains; no visibility decision was auto-approved.',
            'Only four saved poses, not a broad scene/asset acceptance test; triangle-list support and performance remain bounded.',
            'Diagnostic loading still includes both installed and local core libraries; deployment packaging not certified.'],
        inputs={str(p):base.file_sha256(p) for p in paths}))
    print(run.stderr);print(cpp.stdout);print('FOUR_POSE_TECHNICAL_GATE',passed,'BASELINE_40_PASSED; NO_TRAINING_NO_PRODUCTION_INSTALL')


if __name__=='__main__':main()
