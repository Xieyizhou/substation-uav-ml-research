"""Report the expanded regression failure explicitly; no automatic admission."""
import itertools
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import psutil
from scripts.vision.validate_edge_raster import OUT,BUILD,ROOT,read,verify,frozen,file_sha256,guard
from scripts.vision.run_clip_existing_pose_regression import OUT as POSES,base
from scripts.vision.test_body_material_applicability import bounds,baseline_verify


def compare_pose(control,clipped):
    if any(r['status']!='existing_pose_technical_checks_passed' or not r['process_cleanup_complete'] or len(r['records'])!=3 for r in (control,clipped)):
        raise ValueError('Incomplete or invalid replay')
    results=[]
    for a,b in zip(control['records'],clipped['records']):
        if not a['rgb_exact'] or not b['rgb_exact'] or max(a['skew_ms'],b['skew_ms'])>33.334001:raise ValueError('RGB/time invalid')
        if a['mask_sha256']!=b['mask_sha256']:raise ValueError('Instance mask changed')
        old,new=a['full_boxes'],b['full_boxes']
        deltas={k:max(abs(x-y) for x,y in zip(old[k],new[k])) for k in set(old)&set(new)}
        results.append(dict(lost=sorted(set(old)-set(new)),added=sorted(set(new)-set(old)),
            deltas=deltas,over_one_pixel={k:d for k,d in deltas.items() if d>1},
            control_boxes=old,clipped_boxes=new))
    held=any(r['lost'] or r['added'] or r['over_one_pixel'] for r in results)
    return dict(status='held_existing_box_change' if held else 'existing_box_gate_passed',stable_comparisons=results)


def geometry_depth(frame,label):
    tree=base.ET.parse(frame['source_world']);name=frame['instance_mapping'][label]['object_id']
    model=tree.find(f".//world/model[@name='{name}']")
    def pose(node):
        values=np.array(list(map(float,node.findtext('pose','0 0 0 0 0 0').split())))
        if len(values)!=6 or np.any(values[3:]) or (node.find('pose') is not None and node.find('pose').attrib):raise ValueError('Unsupported transform')
        return values[:3]
    r=np.column_stack([base.rotate(frame['actual_pose']['orientation'],np.eye(3)[i]) for i in range(3)])
    link=tree.find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    optical=np.array(frame['actual_pose']['position'])+r@pose(link)
    sensor=link.find("sensor[@name='research_boxes']");near=float(sensor.findtext('camera/clip/near'));far=float(sensor.findtext('camera/clip/far'))
    hfov=float(sensor.findtext('camera/horizontal_fov'));vertices=[];extrema=[]
    for part in model.findall('link'):
        for visual in part.findall('visual'):
            lo,hi=bounds(visual)
            xyz=np.array(list(itertools.product(*zip(lo,hi))))+pose(model)+pose(part)
            camera=(xyz-optical)@r;vertices.extend(camera)
            ndc=np.c_[-camera[:,1]/camera[:,0]/math.tan(hfov/2),camera[:,2]/camera[:,0]/(math.tan(hfov/2)*1080/1920)]
            pixels=(ndc*np.array([1,-1])+1)*[960,540];index=pixels[:,1].argmax()
            extrema.append(dict(visual=visual.get('name'),lowest_aabb_corner_pixel=pixels[index].tolist(),geometry_type=visual.find('geometry')[0].tag))
    a=np.array(vertices)
    return dict(object_id=name,label=label,depth_interval=[float(a[:,0].min()),float(a[:,0].max())],near=near,far=far,
        all_visual_bounds_between_depth_planes=bool(a[:,0].min()>near and a[:,0].max()<far),
        projected_extrema=extrema,basis='Saved box/cylinder bounding corners; conservative depth bound. Cylinder projected AABB extrema are not exact mesh extrema.')


def main():
    guard();paths=[OUT/'raster-analysis.json',OUT/'visual-review.json',POSES/'protocol.json',POSES/'run-receipt.json',BUILD/'completion.json']
    for p in paths:verify(read(p))
    if read(POSES/'run-receipt.json')['status']!='runs_complete':raise ValueError('Not all six runs complete')
    reports=[];depth=[]
    for frame in read(POSES/'protocol.json')['frames']:
        tag=frame['review_ids'][0];receipts=[]
        for variant in ('control','clipped'):
            rp=POSES/variant/'replay'/tag/'attempt-01/receipt.json';verify(read(rp));receipts.append(read(rp));paths.append(rp)
        comparison=compare_pose(*receipts);reports.append(dict(review_id=tag,**comparison))
        for label in comparison['stable_comparisons'][0]['over_one_pixel']:depth.append(dict(review_id=tag,**geometry_depth(frame,label)))
    # Explicitly preserve residual ownership, rather than silently rounding it away.
    mp=BUILD/'prefix-isolated/clipped/replay/T027/attempt-01/frame-2-mask.bin'
    mask=np.frombuffer(mp.read_bytes(),dtype='u1').reshape(1080,1920,3)
    residual=dict(x=94,y=1075,actual_panoptic_channels=mask[1075,94].tolist(),actual_instance='west_switchgear_03',actual_label=139,
        explanation='Near-boundary ownership disagreement between analytical body-only coverage and rendered instance mask. Precision/coverage convention or visibility competition remains unproven.')
    if residual['actual_panoptic_channels']!=[1,0,139]:raise ValueError('Residual ownership changed')
    tests=['tests.test_edge_raster_validation','tests.test_near_clip_build','tests.test_full_box_runtime_trace','tests.test_full_box_projection','tests.test_dual_box_diagnosis','tests.test_edge_box_trace','tests.test_instance_visibility_diagnosis']
    run=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if run.returncode:raise ValueError(run.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity failed')
    if any(p.info['exe']==str(BUILD/'diagnostic-server') for p in psutil.process_iter(['exe'])):raise ValueError('Diagnostic server remains alive')
    paths += [Path(__file__),mp,ROOT/'docs/results/ml_edge_raster_validation_20260910.md']
    paths += [ROOT/(t.replace('.','/')+'.py') for t in tests]
    dest=OUT/'completion.json'
    if dest.exists():verify(read(dest));print('VALID_EDGE_VALIDATION_REUSED');return
    frozen(dest,dict(status='verification_complete_six_plane_prototype_held_by_regression',
        pose_comparisons=reports,depth_plane_checks=depth,residual_pixel=residual,
        established='Pixel-center sampling reproduces instance128 visible bbox. Expanded same-pose A/B reveals >1px old-box shrinkage at all three other fixed poses; no instance membership loss.',
        next_step='Design a narrower depth-plane-only clipping versus six-plane clipping comparison to preserve established full_2d viewport-box semantics. Do not install current prototype or change labels.',
        production_ready=False,training_ready=False,training_started=False,historical_labels_changed=False,
        baseline=baseline,regression=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(run.stderr);print('VERIFICATION_COMPLETE; SIX_PLANE_PROTOTYPE_HELD; BASELINE_40_PASSED')


if __name__=='__main__':main()
