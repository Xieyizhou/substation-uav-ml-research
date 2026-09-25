"""Finalize only existing reviewed diagnostic evidence; never infer or train."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
from scripts.vision.structure_fit import OUT,ROOT,read,verify,verify_tree,frozen,file_sha256,iou
from scripts.vision.check_structure_fit_sources import strict_review,strict_unit
from scripts.vision.exposure_order_retention import baseline_verify

TESTS = ['tests.test_structure_fit','tests.test_structure_exposure','tests.test_structure_fit_integrity',
    'tests.test_order_retention','tests.test_order_finalization','tests.test_order_protocol_repair',
    'tests.test_retention450_adapter','tests.test_retention_real_quota','tests.test_retention_regression_review','tests.test_canonical_shutdown']
REPORT = ROOT/'docs/results/ml_structure_fit_diagnosis_20260909.md'


def finish():
    if (OUT/'completion.json').exists():
        verify_tree(OUT/'completion.json');print('REUSED_HASH_VALID_COMPLETE_DIAGNOSIS_WITH_GAPS');return
    verify_tree(OUT/'protocol.json')
    print('INPUT_GRAPH_VERIFIED',flush=True)
    files=['protocol.json','evidence.json','review.json','source-audit.json','member-source-trace.json','fp-structure-map.json','analysis.json','diagnostic-matrix.json','publication-repair.json']
    for name in files:verify(read(OUT/name))
    p=read(OUT/'protocol.json');review=read(OUT/'review.json');matrix=read(OUT/'diagnostic-matrix.json')
    strict_review(read(OUT/'evidence.json'),review['decisions'])
    models={};proofs=[]
    for key in p['models']:
        m=read(OUT/'inference'/f'{key}.json');strict_unit(m,key,p);models[key]=m
        for cohort in ('pool','development'):
            for row in m[cohort]:
                for miss in row['misses']:
                    if not miss['formal_matching_competition']:continue
                    t=row['truth'][miss['truth_index']];assign={a['prediction_index']:a for a in row['matches']};competition=[]
                    for i,pred in enumerate(row['predictions']):
                        if pred['class_name']!=t['class_name'] or iou(pred['bbox_xyxy'],t['bbox_xyxy'])<.5:continue
                        a=assign.get(i)
                        if not a or a['truth_index']==miss['truth_index']:raise ValueError('Unexplained matching competition')
                        competition.append(dict(prediction=pred,overlap_with_missed_truth=iou(pred['bbox_xyxy'],t['bbox_xyxy']),
                            actual_assignment=a,assigned_truth=row['truth'][a['truth_index']]))
                    if not competition:raise ValueError('Empty competition proof')
                    proofs.append(dict(model=key,cohort=cohort,member_or_view_id=row.get('member_id',row.get('view_id')),variant=row.get('variant'),
                        missed_truth=t,miss=miss,competition=competition,explanation='Same-class retained prediction already assigned to a different truth by the frozen descending-IoU one-to-one matcher; no double counting.',
                        original_operational_reason_retained=True,independent_instances_added=0))
    if len(proofs)!=len(matrix['formal_matching_competition_events']):raise ValueError('Competition scope differs')
    ip={str(OUT/'diagnostic-matrix.json'):file_sha256(OUT/'diagnostic-matrix.json'),str(Path(__file__)):file_sha256(Path(__file__))}
    for key in models:ip[str(OUT/'inference'/f'{key}.json')]=file_sha256(OUT/'inference'/f'{key}.json')
    frozen(OUT/'matching-competition.json',dict(status='all_matching_competition_mechanically_explained_not_semantic_certification',events=proofs,inputs=ip))
    result=subprocess.run([sys.executable,'-m','unittest',*TESTS],cwd=ROOT,capture_output=True,text=True,timeout=60)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if result.returncode or not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:
        raise ValueError('Verification failed: '+result.stdout+result.stderr)
    if 'Ran 59 tests' not in result.stderr:raise ValueError('Unexpected regression test count')
    tests={str(ROOT/(x.replace('.','/')+'.py')):file_sha256(ROOT/(x.replace('.','/')+'.py')) for x in TESTS}
    tests.update({str(Path(__file__)):file_sha256(Path(__file__)),str(ROOT/'config/perception/visual_experiment_baseline_v1.json'):file_sha256(ROOT/'config/perception/visual_experiment_baseline_v1.json')})
    frozen(OUT/'verification.json',dict(status='related_regressions_and_pinned40_passed',command=[sys.executable,'-m','unittest',*TESTS],
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,related_tests_passed=59,baseline=baseline,full_repository_tests_run=False,inputs=tests))
    if (len(p['negative']),len(p['reactors']),len(p['dev_reactors']),len(models),len(review['decisions']))!=(120,35,10,10,165):raise ValueError('Incomplete diagnosis scope')
    source=read(OUT/'member-source-trace.json');fm=read(OUT/'fp-structure-map.json');a=read(OUT/'analysis.json')
    if len(source['rows'])!=236 or len(fm['events'])!=42:raise ValueError('Incomplete source or FP mapping')
    if (OUT/'inference.lock').exists():raise ValueError('Inference is still active')
    inventory={str(path):file_sha256(path) for path in OUT.rglob('*') if path.is_file() and path.name!='completion.json'}
    inventory[str(REPORT)]=file_sha256(REPORT);inventory[str(Path(__file__))]=file_sha256(Path(__file__))
    for name in ('check_structure_fit_sources.py','map_structure_fit.py','structure_fit.py','run_structure_fit.py','recover_structure_fit.py','review_structure_fit.py','finalize_structure_fit.py'):
        path=ROOT/'scripts/vision'/name;inventory[str(path)]=file_sha256(path)
    frozen(OUT/'completion.json',dict(status='diagnosis_complete_with_named_gaps',all_issues_resolved=False,training_started=False,new_collection=False,labels_modified=False,
        candidate_selected=None,inference_units_complete=10,pool_model_combinations=2360,review_decisions=165,development_FP_events=42,
        named_gaps=dict(unknown_structure_slots=a['unknown_structure_slots'],fine_edge_FP_events=fm['unknown_FP_ROI_events'],
            historical_metadata_members=[r['member_id'] for r in source['rows'] if r['gaps']],
            insufficient_reactor_events=[r['reactor']['event_id'] for r in source['rows'] if r.get('reactor',{}).get('identifiable_content')=='insufficient_or_uncertain'],
            truncation_metadata_disagreements=[r['reactor']['event_id'] for r in source['rows'] if r.get('reactor',{}).get('possible_metadata_erratum')]),
        next_priority='Design an independent supervision-risk revision proposal; do not implement label changes or train.',
        inputs=inventory))
    print('DIAGNOSIS_COMPLETE_WITH_NAMED_GAPS; NO_TRAINING; PINNED40_PASSED; TESTS59_PASSED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    if args.finalize:finish()
    else:print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
