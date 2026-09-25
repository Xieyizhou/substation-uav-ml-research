"""Close one isolated renderer repair test, without promoting code or data."""
import difflib
import subprocess
import sys
from pathlib import Path
import numpy as np
import psutil
from scripts.vision.run_near_clip_build_test import OUT,ROOT,read,verify,frozen,file_sha256,guard
from scripts.vision.test_body_material_applicability import baseline_verify


def compare(control,clipped):
    for receipt in (control,clipped):
        if receipt['status']!='isolated_replay_verified' or not receipt.get('process_cleanup_complete') or len(receipt['records'])!=3:
            raise ValueError('Replay incomplete or cleanup failed')
        if not all(row['rgb_exact'] and row['skew_ms']<=33.334001 for row in receipt['records']):
            raise ValueError('RGB or synchronization gate failed')
    differences=[]
    for a,b in zip(control['records'],clipped['records']):
        old,new=a['full_boxes'],b['full_boxes']
        if set(new)-set(old)!={'128'} or set(old)-set(new):raise ValueError('Unexpected membership change')
        delta={label:max(abs(x-y) for x,y in zip(old[label],new[label])) for label in old}
        if max(delta.values())>1:raise ValueError('Existing box drift requires investigation')
        if a['mask_128_bbox']!=b['mask_128_bbox'] or a['mask_128_pixels']!=b['mask_128_pixels']:raise ValueError('Mask changed')
        if b['mask_128_pixels']!=3534:raise ValueError('Unexpected instance evidence')
        box=new['128'];visible=b['visible_128']['half_open']
        if not all(np.isfinite(box)) or not (box[0]<box[2] and box[1]<box[3]):raise ValueError('Invalid recovered box')
        differences.append(dict(original_box_deltas=delta,recovered_full_box=box,visible_mask_box=visible,
            full_minus_visible=[x-y for x,y in zip(box,visible)],
            maximum_existing_delta=max(delta.values())))
    return differences


def main():
    guard()
    paths=[OUT/'control-build.json',OUT/'protocol.json']
    for p in paths:verify(read(p))
    reports=[]
    for variant in ('control','clipped'):
        folder=OUT/'prefix-isolated'/variant
        inputs=[folder/'protocol.json',folder/'completion.json',folder/'replay/T027/attempt-01/receipt.json']
        for p in inputs:verify(read(p))
        paths+=inputs;reports.append(read(inputs[-1]))
        for library in reports[-1]['loaded_libraries']:
            p=Path(library)
            if not p.is_file():raise ValueError('Missing loaded dependency')
            paths.append(p)
    # Preserve both rejected loader launches as named setup failures.
    rejected=[]
    for name in ('runs','verified-loader'):
        rp=OUT/name/'control/replay/T027/attempt-01/receipt.json'
        verify(read(rp));paths.append(rp)
        rejected.append(dict(path=str(rp),status=read(rp)['status'],reason=read(rp)['reason']))
    differences=compare(*reports)
    source=OUT/'upstream';patched=OUT/'patched-source'
    changed=[]
    for name in read(OUT/'control-build.json')['inputs']:
        p=Path(name)
        if not p.is_relative_to(source):continue
        relative=p.relative_to(source);other=patched/relative
        if not other.is_file():raise ValueError('Patched source missing '+str(relative))
        paths.append(other)
        if file_sha256(p)!=file_sha256(other):changed.append(str(relative))
    if changed!=['ogre2/src/Ogre2BoundingBoxCamera.cc']:
        raise ValueError('Unexpected source changes: '+str(changed))
    for process in psutil.process_iter(['pid','exe']):
        if process.info['exe']==str(OUT/'diagnostic-server'):
            raise ValueError('Diagnostic server still running')
    diff=''.join(difflib.unified_diff((source/changed[0]).read_text().splitlines(True),(patched/changed[0]).read_text().splitlines(True),fromfile='control/'+changed[0],tofile='clipped/'+changed[0]))
    cpp=subprocess.run([str(OUT/'test-clip')],capture_output=True,text=True,timeout=10)
    if cpp.returncode:raise ValueError(cpp.stderr)
    tests=['tests.test_near_clip_build','tests.test_full_box_runtime_trace','tests.test_full_box_projection','tests.test_dual_box_diagnosis','tests.test_edge_box_trace','tests.test_instance_visibility_diagnosis']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity failed')
    partial=OUT/'gz-rendering-8.2.3.tar.bz2'
    paths += [Path(__file__),ROOT/'docs/results/ml_renderer_near_clip_test_20260910.md',partial]
    paths += [ROOT/(t.replace('.','/')+'.py') for t in tests]
    dest=OUT/'completion.json'
    if dest.exists():verify(read(dest));print('VALID_BUILD_TEST_COMPLETION_REUSED');return
    frozen(dest,dict(status='isolated_clip_prototype_recovers_missing_box_not_production_ready',
        source_changes=changed,source_diff=diff,comparisons=differences,rejected_loader_attempts=rejected,
        source_acquisition=dict(used='Git commit f1249da6e07f0d6b7adaba28109fa7af727fd322',
            unused_partial_archive=str(partial),partial_archive_bytes=partial.stat().st_size,
            failure='Release archive download timed out with curl exit 28; partial archive never extracted or used. Interrupted oversized sparse fetch excluded tutorial images; only source checkout used.'),
        standalone_clip_checks=dict(count=9,returncode=cpp.returncode,stdout=cpp.stdout),
        regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),baseline=baseline,
        production_library_written=False,historical_labels_changed=False,training_ready=False,training_started=False,
        unknowns=['Recovered full box left edge differs from visible mask by about 4.24 px. Full geometry and visible extrema need not coincide, but this difference is not yet semantically audited.',
            'Only T027 real-scene comparison; synthetic clipping tests are not multi-scene renderer acceptance.',
            'Diagnostic adapter supports triangle-list FLOAT3/HALF4 meshes with 16/32-bit indices; other topologies fail explicitly.',
            'Both isolated plugin runs load installed and locally built core dylibs; matching RGB/control outputs support this comparison, not a deployment-safe packaging claim.',
            'Performance not benchmarked; polygon clipping changes semantics at viewport boundaries and needs wider regression.'],
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr);print(cpp.stdout);print('ISOLATED_REPAIR_TEST_COMPLETE; PRODUCTION_AND_TRAINING_HELD; BASELINE_40_PASSED')


if __name__=='__main__':main()
