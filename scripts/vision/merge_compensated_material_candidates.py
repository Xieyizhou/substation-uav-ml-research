"""Retain 36 paired images and add one explicitly reviewed common source."""
from pathlib import Path
from scripts.vision.merge_material_pose_candidates import OUT as BASE,prior
from scripts.vision.verify_compensating_pose_pilot import OUT as COMP

OUT=BASE.parent/'compensated-material-candidates-v1'


def main():
    bp=BASE/'reviewed-completion.json';cp=COMP/'reviewed-completion.json'
    b=prior.read(bp);c=prior.read(cp)
    prior.verify(b);prior.verify(c)
    if len(b['members'])!=36 or c['status']!='one_compensating_source_verified_not_training_ready':raise ValueError('Incomplete reviewed input')
    s=c['sources'][0]
    member=dict(member_id='C01-original',pair_id=s['lineage_id'],variant='original',image_path=s['image_path'],image_sha256=s['image_sha256'],
        full_truth=s['full_truth'],instance_mapping=s['instance_mapping'],actual_pose=s['actual_pose'],
        role='common_compensation_member_V_and_VM',source_review=str(cp),training_admitted=False,promotable=False)
    members=b['members']+[member]
    if len({m['member_id'] for m in members})!=37:raise ValueError('Duplicate ID')
    OUT.mkdir(exist_ok=True)
    prior.frozen(OUT/'reviewed-completion.json',dict(status='37_reviewed_candidates_pending_isolation_and_preflight',members=members,
        decisions=b['decisions']+c['decisions'],paired_material_groups=12,common_compensation_images=1,
        training_ready=False,training_started=False,historical_admission_chain_valid=False,
        limitations=['36 original paired images retained; C01 is additional common compensation, not a thirteenth material triplet',
        'Shared complex layout/assets, not independent scenes','Historical chain caveats and final source-role checks unresolved'],
        inputs={str(p):prior.file_sha256(p) for p in (bp,cp,Path(__file__))}))
    print('MERGED37; NO_TRAINING')


if __name__=='__main__':main()
