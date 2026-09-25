"""Bind the analytic result and human-readable report; no new decisions."""
from collections import defaultdict
from pathlib import Path
from scripts.vision.test_body_material_applicability import OUT,ROOT,read,verify,frozen,file_sha256


def main():
    dp=OUT/'diagnosis.json';d=read(dp);verify(d)
    dest=OUT/'completion.json'
    if dest.exists():verify(read(dest));print('VALID_COMPLETION_REUSED');return
    materials=defaultdict(lambda:dict(member_ids=set(),exposures={str(s):0 for s in (7,17,27)}))
    for row in d['training']:
        if row['class_name']!='capacitor_bank':continue
        key=row['signature']['materials']['body']['diffuse'];m=materials[key]
        m['member_ids'].add(row['member_id'])
        for seed,count in row['actual_exposures'].items():m['exposures'][seed]+=count
    for m in materials.values():m['member_ids']=sorted(m['member_ids'])
    report=ROOT/'docs/results/ml_body_material_applicability_20260910.md'
    frozen(dest,dict(status='analytic_test_complete_rendering_and_training_not_started',
        diagnosis_identity=d['identity'],capacitor_body_material_exposure=dict(materials),
        training_ready=False,training_started=False,replay_started=False,new_inference_started=False,
        prior_visual_unknowns_resolved=False,original_training_admission_changed=False,
        next_step='Separately freeze asset-visibility rendering diagnosis, with unchanged replay control first; no historical asset/label edits.',
        inputs={str(p):file_sha256(p) for p in (dp,report,Path(__file__))}))
    print('ANALYTIC_TEST_COMPLETE; NO_TRAINING_OR_RENDERING')


if __name__=='__main__':main()
