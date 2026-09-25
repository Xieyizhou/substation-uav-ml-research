"""Bounded diagnostic handoff; consumes reviews, never approves training."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
from scripts.vision.material_control_feasibility import OUT,RUN,prior
from scripts.vision.verify_experiment_baseline import verify as baseline_verify
from scripts.vision.record_material_feasibility_review import validate_decisions

def neutral(instance):
    values=[x.get('diffuse') for name,x in instance['materials'].items() if name in ('body','reactor')]
    return any(v and list(map(float,v.split()))==[.35,.35,.35,1.] for v in values)

def main():
    names=['initial-gate','risk-containment','independent-review','recomputed-metrics','coverage-census']
    paths=[OUT/(n+'.json') for n in names];e,q,r,m,c=map(prior.read,paths)
    for x in (e,q,r,m,c):prior.verify(x)
    validate_decisions(e['corrected_target_records'],r['decisions'])
    matrix=[]
    for category in ('capacitor_bank','reactor','switchgear','transformer'):
        matches=[(member,t) for member in c['members'] for t in member['instances'] if t['class_name']==category and neutral(t)]
        active=[(member,t) for member,t in matches if any(member['actual_exposures'].values())]
        matrix.append(dict(category=category,gray_instance_exposure={k:sum(member['actual_exposures'][k] for member,t in matches) for k in m['cells']},
            active_member_ids=sorted({member['member_id'] for member,t in active}),
            active_pose_lineages=sorted({member['lineage_id'] for member,t in active}),
            altered_candidate_groups=sorted({x['source_review_id'] for x in c['candidates'] if x['status']=='reviewed_candidate_only' and x['altered_target_class']==category}),
            development_failures=[x for x in m['events'] if x['variant']=='material' and x['category']==category and not x['hit']],
            known_source_condition_coverage=True,sufficient_coverage_certified=False,
            missing_evidence='Current six weights predictions on their actually exposed training members, with own full-label content review.'))
    suites=['tests.test_material_risk_continuation','tests.test_material_control_feasibility','tests.test_closed_material_review',
            'tests.test_canonical_gates','tests.test_exposure_diagnosis','tests.test_paired_visual_factors','tests.test_pixel_duplicates']
    test=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True,timeout=60)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    plan=dict(priority='existing_weight_training_fit_diagnosis',training_authorized=False,new_capture_authorized=False,
        weights=list(m['cells']),initialization_comparator='v2.11, report zero training exposure',
        first_members=sorted({mid for row in matrix for mid in row['active_member_ids']}),
        expansion='Their resolved same-pose original/background training members only; zero-exposure members reported separately.',
        protocol=dict(device='cpu',imgsz=640,confidence=[.37,.001],nms_iou=.7,agnostic_nms=False,max_det=300,matching_iou=.5),
        review='Before interpreting fit, inspect every complete label in these members using own correct crops; reuse only exact hash-valid explicit decisions. Unknown is not valid supervision.',
        decision='Training errors on reviewed exposed samples support fit instability; training success plus development failure supports condition-transfer limitation. Neither proves unique mechanism.',
        hold_rule='T027/T036 resolved source groups remain zero; no restoration, relabeling, training or new material collection in this handoff.')
    paths += [Path(__file__).resolve(),prior.ROOT/'scripts/vision/record_material_feasibility_review.py',
        prior.ROOT/'scripts/vision/continue_material_feasibility_metrics.py',prior.ROOT/'scripts/vision/material_feasibility_coverage_census.py',
        prior.ROOT/'scripts/vision/contain_material_candidate_risks.py']
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'continuation-completion.json',dict(status='bounded_feasibility_diagnosis_complete_with_named_gaps',
        candidate_selected=None,training_ready=False,training_started=False,risks_contained_not_repaired=True,
        reviewed_truths=len(r['decisions']),visual_unknowns=r['unknowns'],coverage_matrix=matrix,next_stage=plan,
        limitations=['12 pose groups, not 1440 independent examples','65 positive records lack contemporaneous world/mapping attestations; saved-source reconstruction only',
            '120 negative worlds not part of target material census; full labels and export pixels checked',
            'Same complex layout/assets are not source-independent; no sufficiency threshold inferred',
            'No exhaustive new full-frame missing-label audit of the entire current training pool'],
        tests_output=test.stdout+test.stderr,baseline=b,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('BOUNDED_DIAGNOSIS_COMPLETE; NEXT: EXISTING_WEIGHT_TRAINING_FIT; NO TRAINING')

if __name__=='__main__':main()
