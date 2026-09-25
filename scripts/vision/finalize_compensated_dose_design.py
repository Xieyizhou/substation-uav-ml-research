"""Freeze evidence-based decision and unchanged evaluation gates; no training."""
from collections import Counter
from pathlib import Path
from scripts.vision.freeze_compensated_double_dose import OUT,prior,source,checks,vector
from scripts.vision.compensated_pool_fit import OUT as FIT
from scripts.vision.build_compensated_error_review import DEST
from scripts.vision.train_frozen_multiscale import contract as reference_contract


def main():
    pp=OUT/'protocol.json';p=prior.read(pp);prior.verify(p)
    old=prior.read(source.OUT/'protocol.json');prior.verify(old);checks(old,p)
    fit=prior.read(FIT/'summary.json');prior.verify(fit)
    review=prior.read(DEST/'review-summary.json');prior.verify(review)
    paths=[pp,FIT/'summary.json',DEST/'review-summary.json',OUT/'counts.json',DEST/'reviewed-dose-control-v1/counts.json',Path(__file__).resolve()]
    vectors={(1,1,0,1,0),(1,2,0,1,0),(1,1,2,0,1),(1,1,3,0,1)}
    proof=[]
    for s in (7,17,27):
        counts=Counter(old['schedules'][f'V-{s}']);oldrows=[r for r in old['pool_rows'] if 'full_truth' not in r and r['subset']=='bridge_positive']
        if {tuple(vector(r)) for r in oldrows if counts[r['member_id']]}-vectors:raise ValueError('Separating identity no longer applicable')
        available=sum(counts[r['member_id']] for r in oldrows if tuple(vector(r))==(1,1,0,1,0))
        # Additional triple-dose demand: E=108,T=132,C=42 -> b=T-E=24,a=C-b=18.
        if available>=18:raise ValueError('Claimed threefold obstruction invalid')
        proof.append(dict(seed=s,required_vector_1_1_0_1_0=18,available_even_if_all_removed=available,
            deficit=18-available,independent_of_minimum_one_old_member_constraint=True,
            derivation='Old bridge patterns a=(1,1,0,1,0), b=(1,2,0,1,0), c=(1,1,2,0,1), d=(1,1,3,0,1). Additional E=108,T=132,C=42 forces b=24,a=18.'))
    _,policy,_=reference_contract('fixed-7')
    paths.append(Path(reference_contract('fixed-7')[0]['source_protocol']))
    fit_table=[]
    for seed in (7,17,27):
        material=[x for x in fit['instances'] if x['model']==f'VM-{seed}' and x['new_compensated_member'] and x['actual_exposures']>0 and x['variant'] in ('warm','cool') and x['truth']['class_name']=='capacitor_bank']
        fit_table.append(dict(seed=seed,material_capacitor_truth=len(material),material_capacitor_hit=sum(x['hit'] for x in material),
            actual_member_exposures=sorted({x['actual_exposures'] for x in material})))
    return prior.frozen(OUT/'decision-protocol.json',dict(status='frozen_bounded_double_dose_comparison',
        primary_question='At fixed 450-step total budget and full-class instance totals, does shifting more old bridge exposure to the same reviewed new pose/material groups improve native material fit without unacceptable development retention loss?',
        confirmed_facts=fit_table,threefold_infeasibility_certificate=proof,
        interpretation='Twofold is the largest feasible integer uniform multiplier under the frozen bridge-only substitution and exact class constraints; it is not a sufficient-learning threshold.',
        experimental_units=[dict(cell=k,seed=int(k.split('-')[-1]),arm=k.split('-')[0],steps=450,exposures=2700,new_member_exposures=108) for k in source.KEYS],
        initialization='independent_v2.11_not_previous_endpoint',
        direct_controls={k:dict(training=str(source.OUT/'training'/k/'completion.json'),evaluation=str(source.OUT/'evaluation'/f'{k}.json')) for k in source.KEYS},
        comparisons=['V108 vs historical V54','VM108 vs historical VM54','VM108 vs V108','Appearance effect change between 54 and 108; descriptive interaction, not independent scenes'],
        acceptance_policy=policy['acceptance_policy'],retention=policy['retention'],
        evaluation=p['evaluation'],
        evaluation_requirements=['48 paired plus48 no-target already-viewed development frames','Formal .37 and separate .001; CPU640,class-aware NMS .7,max300,one-to-one same-class IoU>=.5',
            'All three seeds and endpoint last.pt only; no threshold search','All new false positives and original/lighting losses reviewed; gains and persistent outcomes retained',
            'Training-fit diagnosis on the same frozen277 member universe; unexposed separate','Full metrics and original/light per-class retention remain decisive; no filtering18 uncertain targets',
            'Match conflicts, missing/stale audit or outputs block candidate selection'],
        outcome_rules=['Native material fit and development improve with retention: supports further testing this substitution strategy, not unique cause',
            'Native fit improves but development does not: prioritize condition-transfer diagnosis',
            'Native material fit remains poor: next controlled optimization comparison; no automatic extra exposure',
            'Capability tradeoff or seed inconsistency: report all seeds; no promotion or automatic follow-on experiment'],
        limits=['No old member increases or previously zero member restoration','New37 data reviewed113 labels; old whole pool not recertified',
            'Same 13 source poses, same assets/layout; no new independent scenes','C01/S07 within-new-group concentration unchanged; total new-group concentration doubles',
            '54 extra changed member/brightness combinations; old bridge suppression is inseparable from fixed-budget dose effect',
            'Native fit is not augmentation-tensor fit or generalization','Development18 content-boundary gaps are retained, not repaired labels'],
        training_started=False,automatic_training_authorized=False,automatic_follow_on_experiment=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
