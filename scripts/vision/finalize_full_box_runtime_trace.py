"""Verify runtime branch evidence; never admit labels or start training."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
from scripts.vision.verify_full_box_runtime_trace import FIRST,OUT,PLUGIN,PLUGIN_HASH,guard
from scripts.vision.run_full_box_runtime_trace import ROOT,read,verify,frozen,file_sha256,DUAL,SOURCE,ANALYTIC
from scripts.vision.diagnose_full_box_projection import reconstruct,SIGNS
from scripts.vision.test_body_material_applicability import baseline_verify,model_mapping
from scripts.vision.instance_visibility_diagnosis import ET


def validate_trace(events, frame):
    expected=['library_loaded']+['post_mesh_projection','branch_observed']*3+['trace_complete']
    if [e.get('event') for e in events]!=expected:
        raise ValueError('Missing, duplicate, failed or incomplete trace events')
    library=events[0]
    if Path(library['path']).resolve()!=PLUGIN.resolve() or library['uuid']!='8FBF722A-207A-3417-B576-D1D9F28C9BC3':
        raise ValueError('Wrong loaded plugin identity')
    if events[-1]['observed_branches']!=3:
        raise ValueError('Incomplete branch observations')
    times=[e['monotonic'] for e in events]
    if times!=sorted(times) or len(times)!=len(set(times)):
        raise ValueError('Invalid event chronology')
    mapping,models=model_mapping(ET.parse(frame['source_world']))
    if mapping.get('128')!='west_switchgear_02' or frame['instance_mapping'].get('128') is None:
        raise ValueError('Instance mapping conflict')
    analytic=reconstruct(frame)
    body=next(v for v in analytic['visuals'] if v['visual']=='body')
    signatures=[];observations=[]
    for start in (1,3,5):
        p,b=events[start:start+2]
        if p['label']!=128 or b['label']!=128 or p['item_id']!=b['item_id']:
            raise ValueError('Trace instance mismatch')
        if b['branch']!='rejected' or b['instruction_offset']!=632:
            raise ValueError('Rejection branch not observed')
        for key,want in (('position',[-13,-3.5,.8]),('scale',[1.72,2.58,1.44]),('orientation_wxyz',[1,0,0,0])):
            if not np.allclose(p[key],want,rtol=0,atol=1e-6):
                raise ValueError('Component transform conflict')
        low=np.asarray(p['min_vertex']);high=np.asarray(p['max_vertex'])
        if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all():
            raise ValueError('Invalid projection extrema')
        if not (abs(low[0])>1 and abs(high[0])>1 and low[0]<-1<1<high[0]):
            raise ValueError('Cross-viewport x rejection not reproduced')
        if not np.allclose([low[:2],high[:2]],body['raw_ndc_bounds'],rtol=0,atol=.001):
            raise ValueError('Analytical/runtime projection conflict')
        view=np.asarray(p['view_matrix']).reshape(4,4);projection=np.asarray(p['projection_matrix']).reshape(4,4)
        if not np.isfinite([view,projection]).all():raise ValueError('Nonfinite matrices')
        center=np.linalg.inv(view)[:3,3]
        if not np.allclose(center,analytic['optical_center'],rtol=0,atol=1e-5):
            raise ValueError('Runtime camera differs from saved optical center')
        # Use actual runtime matrices on saved box corners as an independent check.
        world=np.array([np.asarray(p['position'])+np.asarray(p['scale'])*s/2 for s in np.asarray(SIGNS)])
        clip=(np.c_[world,np.ones(8)]@view.T)@projection.T
        w=clip[:,3];ndc=clip[:,:3]/w[:,None]
        # MeshMinimalBox divides x/y by w but retains projected z undivided.
        if not w.min()<0<w.max() or not np.allclose([ndc[:,:2].min(0),ndc[:,:2].max(0)],[low[:2],high[:2]],rtol=0,atol=.002) or not np.allclose([clip[:,2].min(),clip[:,2].max()],[low[2],high[2]],rtol=0,atol=.002):
            raise ValueError('Runtime matrices do not explain unclipped projection')
        signatures.append(json.dumps({k:p[k] for k in ('item_id','camera_pointer','min_vertex','max_vertex','position','scale','view_matrix','projection_matrix')},sort_keys=True))
        observations.append(dict(item_id=p['item_id'],label=128,component='body',component_basis='unique saved model/visual transform plus visible-map label; not component pixel-mask certification',
            min_vertex=p['min_vertex'],max_vertex=p['max_vertex'],runtime_optical_center=center.tolist(),
            corner_perspective_w_range=[float(w.min()),float(w.max())],branch='rejected',instruction_offset=632))
    if len(set(signatures))!=1:raise ValueError('Unstable runtime observations')
    return observations


def validate_alignment(receipt):
    if receipt['status']!='dual_mode_diagnostic_complete' or not receipt.get('control_rgb_exact') or not receipt.get('process_cleanup_complete'):
        raise ValueError('Replay alignment or cleanup failed')
    comparisons=receipt['comparisons']
    if len(comparisons)!=3:raise ValueError('Missing stable replay comparisons')
    for c in comparisons:
        if 128 in c['full_labels'] or c['instance_128_pixels']!=3534 or c['mask_bbox_xyxy']!=[59,1052,306,1080] or c['visible_boxes']['128']['half_open']!=c['mask_bbox_xyxy'] or c['visible_skew_ms']>33.334001:
            raise ValueError('Dual-mode evidence differs')


def main():
    guard()
    paths=[OUT/'binary-binding.json',OUT/'binary-postcheck.json',OUT/'run-receipt.json',OUT/'protocol.json',
           FIRST/'run-receipt.json',FIRST/'protocol.json',DUAL/'completion.json',DUAL/'protocol.json',
           SOURCE/'protocol.json',ANALYTIC/'diagnosis.json']
    for p in paths:verify(read(p))
    frame=read(OUT/'protocol.json')['frames'][0]
    verify(read(frame['source_receipt']))
    runs=[]
    for root in (FIRST,OUT):
        folder=root/'replay/T027/attempt-01';rp=folder/'receipt.json'
        verify(read(rp));validate_alignment(read(rp))
        trace=folder/'runtime-trace.jsonl'
        observations=validate_trace([json.loads(s) for s in trace.read_text().splitlines()],frame)
        runs.append(dict(directory=str(folder),observations=observations,selected_capture_indices=read(rp)['selected_capture_indices']))
        paths += [rp,trace]
        cleanup=read(root/'run-receipt.json')['debugger_cleanup']
        if len(cleanup)!=1 or cleanup[0]['survivors']:
            raise ValueError('Missing or failed descendant cleanup')
        for pid,created in cleanup[0]['owned']:
            try:
                process=psutil.Process(pid)
                if process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE:
                    raise ValueError('Owned diagnostic descendant still running')
            except psutil.NoSuchProcess:pass
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity failed')
    tests=['tests.test_full_box_runtime_trace','tests.test_full_box_projection','tests.test_dual_box_diagnosis','tests.test_edge_box_trace','tests.test_instance_visibility_diagnosis']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    paths += [PLUGIN,Path(__file__),Path(frame['source_world']),Path(frame['source_receipt']),ROOT/'docs/results/ml_full_box_runtime_trace_20260910.md']
    paths += [ROOT/(t.replace('.','/')+'.py') for t in tests]
    dest=OUT/'completion.json'
    if dest.exists():verify(read(dest));print('VALID_RUNTIME_TRACE_COMPLETION_REUSED');return
    frozen(dest,dict(status='runtime_full_2d_rejection_confirmed_for_t027_instance128',runs=runs,
        established='The visible-map-labelled body item reaches MeshMinimalBox, returns crossing-x extrema, then enters the installed FullBoundingBoxes rejection destination. Reproduced twice at the saved optical pose; unchanged RGB/full boxes and mask-consistent visible_2d.',
        identity_correction='Initial library pin referred to top-level dylib. Actual plugin whole-file hash differs; FullBoundingBoxes instruction streams and UUID match. Second run binds both files before and after execution.',
        unresolved=['No repaired renderer evaluated; no claim of a general safe fix.',
            'Trace invocations have no simulator timestamps; three stable sensor comparisons are separate observations in each same-pose replay, not a per-message trace join.',
            'Broader dataset impact and edge-fragment supervision policy unresolved; T036 quality hold unchanged.',
            'No direct raw-VAO dump: runtime MeshMinimalBox extrema and matrices observed; saved-corner reconstruction is additional evidence.'],
        training_ready=False,training_started=False,historical_labels_changed=False,production_library_written=False,
        actual_plugin_sha256=PLUGIN_HASH,baseline=baseline,
        regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr);print('RUNTIME_BRANCH_CONFIRMED; TRAINING_HELD; BASELINE_40_PASSED')


if __name__=='__main__':main()
