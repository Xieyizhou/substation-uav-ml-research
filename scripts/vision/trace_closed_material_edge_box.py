"""Read-only stage trace: raw Gazebo messages versus conversion and visible mask."""
import copy
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
from scripts.vision.closed_body_material_pilot import OUT as SOURCE,ROOT,read,verify,frozen,file_sha256
from scripts.vision.test_body_material_applicability import baseline_verify
from src.vision.collection.gazebo_truth import parse_gazebo_truth_message

OUT=SOURCE/'edge-box-trace-v1'


def parse(message):
    return parse_gazebo_truth_message(message,topic='/research_camera/boxes',width=1920,height=1080,receive_index=1)


def synthetic_box(bounds,label=128):
    a,b,c,d=bounds
    return dict(label=label,box=dict(minCorner=dict(x=a,y=b),maxCorner=dict(x=c,y=d)))


def labels(truth):
    return [int(re.search(r'instance-(\d+)-',x.annotation_id)[1]) for x in truth.objects]


def run():
    OUT.mkdir(exist_ok=True);dest=OUT/'diagnosis.json'
    paths=[SOURCE/p for p in ('protocol.json','review-completion.json','mask-coverage.json','semantic-review.json')]
    for p in paths:verify(read(p))
    if dest.exists():verify(read(dest));print('VALID_STAGE_TRACE_REUSED');return
    frame=next(x for x in read(SOURCE/'protocol.json')['frames'] if x['review_ids']==['T027'])
    sp=Path(frame['source_receipt']);source=read(sp);verify(source);paths.append(sp)
    original=parse(source['raw_truth'])
    if not original.valid:raise ValueError('Source conversion invalid')
    original_labels=labels(original)
    source_saved=[int(re.search(r'instance-(\d+)-',x['annotation_id'])[1]) for x in source['truth']['objects']]
    if sorted(original_labels)!=sorted(source_saved):raise ValueError('Current conversion and source membership disagree')
    rows=[]
    for variant in ('control','warm','cool'):
        folder=SOURCE/'renders/T027'/variant/'replay/T027/attempt-01';rp=folder/'receipt.json';r=read(rp);verify(r);paths.append(rp)
        frames=[]
        for i in range(1,13):
            bp=folder/f'frame-{i}-boxes.json';mp=folder/f'frame-{i}-mask.bin';raw=read(bp);converted=parse(raw)
            if not converted.valid:raise ValueError('Recorded raw frame invalid under converter')
            rawlabels=[int(x['label']) for x in raw.get('annotatedBox',[])];mapped=labels(converted)
            if sorted(rawlabels)!=sorted(mapped):raise ValueError('Conversion changed classifiable raw membership')
            mask=np.frombuffer(mp.read_bytes(),dtype='u1').reshape(1080,1920,3);ys,xs=np.where(mask[:,:,2]==128)
            frames.append(dict(index=i,in_selected_stable_window=i in r['selected_capture_indices'],raw_labels=rawlabels,
                converted_labels=mapped,instance_128_pixels=len(xs),bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None))
            paths.extend((bp,mp))
        rows.append(dict(variant=variant,frames=frames,selected_indices=r['selected_capture_indices']))
    counterfactual=copy.deepcopy(source['raw_truth']);counterfactual['annotatedBox'].append(synthetic_box([59,1052,306,1080]))
    recovered=parse(counterfactual)
    if not recovered.valid or 128 not in labels(recovered):raise ValueError('Edge converter control failed')
    tests=['tests.test_edge_box_trace','tests.test_closed_material_review','tests.test_instance_visibility_diagnosis','tests.test_canonical_gates']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    b=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline mismatch')
    versions=subprocess.check_output(['pkg-config','--modversion','gz-rendering8','gz-sensors8'],text=True).splitlines()
    paths += [Path(__file__),ROOT/'src/vision/collection/gazebo_truth.py',ROOT/'src/vision/canonical/collect.py',
        ROOT/'src/vision/collection/simulator_labels.py',ROOT/'tools/gz_visibility_capture_cleanup_fixed.cc',
        Path(frame['source_plan']),Path(frame['source_world']),Path('/opt/homebrew/opt/gz-rendering8/INSTALL_RECEIPT.json'),Path('/opt/homebrew/opt/gz-sensors8/INSTALL_RECEIPT.json')]
    paths += [ROOT/Path(t.replace('.','/')+'.py') for t in tests]
    frozen(dest,dict(status='missing_instance_localized_upstream_of_python_conversion',source_raw_labels=original_labels,
        source_saved_labels=source_saved,replays=rows,
        synthetic_converter_control=dict(test_only_not_new_annotation=True,bbox_xyxy=[59,1052,306,1080],result_valid=recovered.valid,
            label_128_preserved=128 in labels(recovered)),
        findings=['Instance 128 is absent in source raw message before conversion/export.',
            'Recorded replay raw box messages also lack 128 while aligned equipment masks contain it.',
            'Current converter retains a synthetic positive-area edge box; zero-area or unknown labels invalidate, not silently accept a smaller complete set.',
            'No documented small-area/edge ignore rule was found in the inspected active collection/conversion path. Candidate pose occlusion filtering is not annotation filtering.'],
        upstream_source_review=dict(runtime_versions=versions,
            rendering_url='https://github.com/gazebosim/gz-rendering/blob/gz-rendering8_8.2.3/ogre2/src/Ogre2BoundingBoxCamera.cc',
            sensor_url='https://github.com/gazebosim/gz-sensors/blob/gz-sensors8_8.2.2/src/BoundingBoxCameraSensor.cc',
            assessment='Version-matched upstream full_2d path includes visibility-map filtering, frustum checks and projected-bound filtering before publishing. Exact failing branch for this scene is not instrumented; upstream source inspection is supporting evidence, not a runtime branch trace.'),
        training_started=False,historical_labels_changed=False,automatic_ignore_policy_created=False,
        next_step='Separate same-pose diagnostic comparing full_2d and visible_2d outputs or instrument full_2d filtering; do not switch production mode, patch installed renderer or fabricate original labels.',
        baseline=b,regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr)
    print([(u['variant'],sum(128 in f['raw_labels'] for f in u['frames']),[(f['instance_128_pixels'],f['bbox_xyxy']) for f in u['frames'] if f['in_selected_stable_window']]) for u in rows])


if __name__=='__main__':run()
