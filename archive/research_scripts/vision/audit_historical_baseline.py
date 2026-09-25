"""Audit legacy four-weight paired evaluation without rerunning inference."""
from pathlib import Path
from scripts.vision.audit_historical_results import OUT,CURRENT,prior,integrity,paired_truth,score,summary,VARIANTS

def main():
    old=prior.ROOT/'data/research/ml_training_recovery_v1/paired-visual-factors-v1/evaluation.json'
    r=prior.read(old);cache={};issues=integrity(old,cache)
    p=prior.read(CURRENT/'protocol.json');rp=Path(p['evaluation']['paired_review']);paired,inputs=paired_truth(prior.read(rp)['frames'])
    lookup={(x['view_id'],x['variant']):(x,t) for x,t in paired};models={}
    for name,record in r['results'].items():
        if len(record['rows'])!=48 or {(x['view_id'],x['variant']) for x in record['rows']}!=set(lookup):raise ValueError('Legacy membership differs')
        fresh=[]
        for row in record['rows']:
            src,t=lookup[row['view_id'],row['variant']]
            if t!=row['truth']:raise ValueError('Legacy truth differs')
            fresh.append(score(src,t,row['predictions'],[]))
        sm={v:summary([x for x in fresh if x['variant']==v]) for v in VARIANTS}
        delta={v:{m:(sm[v][m]-record['summary'][v][m] if sm[v][m] is not None and record['summary'][v][m] is not None else None) for m in ('planned_instance_hit_rate','instance_recall','matched_precision','unmatched_predictions')} for v in VARIANTS}
        models[name]=dict(weight_hash_valid=prior.file_sha256(record['weights_path'])==record['weights_sha256'],summary=sm,formal_metric_deltas=delta,
            planned_matching_conflicts=sum(x['matching_conflict'] for x in fresh),
            legacy_by_class_semantics='Groups frames by planned category, then summarizes ALL truth classes in those frames; not class-filtered instance recall.',
            diagnostic_misses_recomputed=False,negative_FPR_available=False)
        inputs[record['weights_path']]=prior.file_sha256(record['weights_path'])
    paths=[old,rp,Path(__file__).resolve()]
    inputs.update({str(x):prior.file_sha256(x) for x in paths})
    prior.frozen(OUT/'baseline-paired-audit.json',dict(status='four_weight_formal_recalculation_complete',legacy_identity_issues=issues,models=models,
        caveat='No low-threshold predictions or no-target frames in this historical artifact; do not infer missing metrics.',inputs=inputs))
    print('BASELINE_AUDITED',len(models),'IDENTITY_ISSUES',len(issues))

if __name__=='__main__':main()
