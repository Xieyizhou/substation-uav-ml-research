"""Independent pre-design validation, not permission to train."""
import subprocess
import sys
from pathlib import Path
from collections import Counter
from scripts.vision.design_visibility_repair_contrast import OUT,SOURCE,ROOT,REFERENCE,read,save,file_sha256,verify_tree,baseline_verify,schedule

def full_labels_equal(first,second):
    if first!=second:raise ValueError('Paired full labels differ; equal class/size is insufficient')

def main():
    dest=OUT/'design-validation.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    verify_tree(OUT/'design.json');r=read(OUT/'design.json');old=read(REFERENCE/'protocol.json')
    lookup={x['member_id']:x for x in old['pool_rows']};trace=read(SOURCE.parent/'trace.json')
    expected={(arm,s) for arm in ('original_only','three_variant') for s in (7,17,27)}
    if set(r['schedules'])!={f'{a}-300-{s}' for a,s in expected}:raise ValueError('Training unit scope changed')
    allowed=set(r['allowed_bridge_members']);texts={mid:Path(row['label_path']).read_text() for mid,row in lookup.items()}
    signatures={f['member_id']:sorted((o['device_id'],o['category'],o['bbox_xyxy']) for o in f['objects']) for f in trace['frames']}
    for seed in (7,17,27):
        baseline=old['schedules'][f'I-300-{seed}'];o=r['schedules'][f'original_only-300-{seed}'];m=r['schedules'][f'three_variant-300-{seed}']
        if (o,m)!=schedule(baseline,old['pool_rows'],r['group_quota'],trace,seed):raise ValueError('Frozen sampling not reproducible')
        for a,b,p in zip(o,m,baseline):
            if lookup[p]['subset']=='bridge_positive':
                if a not in allowed or b not in allowed:raise ValueError('Held member selected')
                full_labels_equal(texts[a],texts[b])
                if signatures[a]!=signatures[b]:raise ValueError('Paired source instance/coordinate conflict')
            elif (a,b)!=(p,p):raise ValueError('Common slot changed')
        for draws in (o,m):
            for block in range(3):
                counts=Counter()
                for mid in draws[block*600:(block+1)*600]:counts.update(lookup[mid]['class_instances'])
                if dict(counts)!={'transformer':525,'switchgear':525,'capacitor_bank':220,'reactor':220}:raise ValueError('Full supervision changed')
    modules=['test_visibility_repair_design','test_visibility_repair_full_labels','test_remaining_bridge_summary','test_retained_bridge_review']
    test=subprocess.run([sys.executable,'-m','unittest',*['tests.'+m for m in modules]],cwd=ROOT,capture_output=True,text=True,check=True)
    report=ROOT/'docs/results/ml_visibility_repair_contrast_design_20260909.md'
    paths=[OUT/'design.json',Path(__file__),report,*[ROOT/'tests'/(m+'.py') for m in modules]]
    save(dest,dict(status='design_validated_training_not_started',all_six_sequences_reproduced=True,
        full_source_instance_and_label_coordinates_equal=True,full_instance_exposures_per_300=dict(transformer=1575,switchgear=1575,capacitor_bank=660,reactor=660),
        regression_output=test.stderr,whole_repository_tested=False,
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
        remaining_gate='Bind evaluation and reference identities, audit inherited common-pool admission, export separate dataset and validate actual sampler before training.',
        inputs={str(p):file_sha256(p) for p in paths}))
    print('DESIGN_VALIDATED_NOT_TRAINED',flush=True)

if __name__=='__main__':main()
