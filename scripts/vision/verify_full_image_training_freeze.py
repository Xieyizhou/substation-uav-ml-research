"""Independent readback of exports and frozen schedules; not a training launcher."""
import subprocess
import sys
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.freeze_full_image_training import OUT, REFERENCE, ROOT, NAMES, SEEDS, read, save, file_sha256, verify_tree, encode, schedules, exposures, baseline_verify

MODULES = ('tests.test_full_image_training_freeze','tests.test_full_image_remaining',
    'tests.test_full_image_appearance','tests.test_full_image_pilot_review',
    'tests.test_full_image_pose_screen','tests.test_full_image_pose_screen_v2',
    'tests.test_appearance_recovery_world','tests.test_canonical_gates','tests.test_canonical_recovery')

def main():
    dest = OUT/'freeze-validation.json'; pp = OUT/'protocol.json'
    if dest.exists(): verify_tree(dest); print('VERIFIED_EXISTING',dest); return
    verify_tree(pp)
    p = read(pp); dataset = read(p['dataset_path']); members = dataset['members']
    if len(members)!=56 or len({r['member_id'] for r in members})!=56:
        raise ValueError('Incomplete or duplicate member list')
    if p['training_admitted'] or p['promotable'] or p['training_started']:
        raise ValueError('Unexpected admission or execution')
    for r in members:
        with Image.open(r['source_image_path']) as a, Image.open(r['image_path']) as b:
            if a.size!=b.size or a.convert('RGB').tobytes()!=b.convert('RGB').tobytes():
                raise ValueError('Pixel readback mismatch')
            if Path(r['label_path']).read_text()!=encode(r['objects'],b.size):
                raise ValueError('Exported labels differ from reviewed full-image boxes')
    for r in p['pool_rows']:
        counts=Counter()
        for line in Path(r['label_path']).read_text().splitlines():
            c,*box=map(float,line.split())
            if c!=int(c) or not 0<=c<len(NAMES) or len(box)!=4:
                raise ValueError('Malformed training class or box')
            counts[NAMES[int(c)]]+=1
        if dict(counts)!=r['class_instances']:
            raise ValueError('Actual label class counts disagree with exposure metadata')
    prior=read(REFERENCE/'protocol.json')
    for seed in SEEDS:
        control,appearance,swaps=schedules(prior,members,seed)
        if swaps!=p['paired_swaps'][str(seed)]:raise ValueError('Swap positions differ')
        for arm,seq in (('K',control),('L',appearance)):
            key=f'{arm}-300-{seed}'
            if p['schedules'][key]!=seq or p['exposures'][key]!=exposures(p['pool_rows'],seq):
                raise ValueError('Frozen schedule or exposure differs')
        if p['exposures'][f'K-300-{seed}']['class_instance_exposure']!=p['exposures'][f'L-300-{seed}']['class_instance_exposure']:
            raise ValueError('Class supervision changed between arms')
    result=subprocess.run([sys.executable,'-m','unittest',*MODULES],cwd=ROOT,capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    paths=[pp,Path(__file__),*(ROOT/(m.replace('.','/')+'.py') for m in MODULES)]
    save(dest,dict(status='data_and_design_frozen_execution_not_started',frame_count=56,box_observations=77,
        pose_count=8,registered_source_groups=7,scene_count=1,
        full_image_labels_readback_verified=True,original_resolution_pixels_verified=True,
        label_class_exposure_verified=True,schedules_verified=6,
        regression=dict(command=[sys.executable,'-m','unittest',*MODULES],returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
        remaining=['Implement and validate execution adapter, including exact actual exposure, optimizer controls, failure attempts and resume checks.',
          'Bind adapter and tests to independent execution receipt before launching six cells.',
          'Apply unchanged development gates plus matched K retention; all three seeds retained; sealed test stays closed.'],
        training_started=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('FREEZE_VALIDATED',dest,flush=True)

if __name__=='__main__':main()
