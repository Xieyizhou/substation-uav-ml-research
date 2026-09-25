"""Symbolic same-source replacements only; quality and actual loader still required."""
from collections import Counter
from pathlib import Path
from scripts.vision import train_compensated_material as source
from scripts.vision.freeze_condition_transfer_probe import OUT as DESIGN, CONDITIONS
from scripts.vision.audit_material_transfer_scope import prior,resolve_truth,CAND


def main():
    pp=source.OUT/'protocol.json';p=prior.read(pp);prior.verify(p)
    dp=DESIGN/'protocol.json';d=prior.read(dp);prior.verify(d)
    cp=CAND/'reviewed-completion.json';c=prior.read(cp);prior.verify(c)
    candidates={m['member_id']:m for m in c['members']}
    rows={r['member_id']:r for r in p['pool_rows']}
    sources={s['source_pose_id']:s for s in d['sources']};results=[]
    for seed in (7,17,27):
        key=f'VM-{seed}';sequence=p['schedules'][key]
        if len(sequence)!=2700 or len(p['brightness_factors'][key])!=2700:raise ValueError('Budget mismatch')
        positions=[]
        for i,mid in enumerate(sequence):
            m=rows[mid]
            if 'full_truth' not in m or m.get('variant') not in ('warm','cool'):continue
            candidate=candidates[mid]
            if candidate['image_sha256']!=m['image_sha256'] or candidate['full_truth']!=m['full_truth']:
                raise ValueError('Candidate provenance mismatch')
            sid=candidate['source_pose_id'];f=sources[sid]['source_frame']
            original=prior.read(f['source_receipt'])['truth']
            a=resolve_truth(m['full_truth'],candidate['instance_mapping']);b=resolve_truth(original,f['instance_mapping'])
            if a!=b:raise ValueError('Source full truth differs; no budget-preserving substitution')
            positions.append(dict(position=i,old_member=mid,source_id=sid,
                class_instances=dict(Counter(x['category'] for x in a)),brightness=p['brightness_factors'][key][i]))
        if len(positions)!=30 or len({x['old_member'] for x in positions})!=24:raise ValueError('Historical exposure changed')
        for condition in CONDITIONS:
            symbolic=list(sequence)
            for x in positions:symbolic[x['position']]=x['source_id']+'-'+condition
            changed={x['position'] for x in positions}
            if any(a!=b for i,(a,b) in enumerate(zip(sequence,symbolic)) if i not in changed):raise ValueError('Unexpected change')
            results.append(dict(seed=seed,condition=condition,symbolic_sequence=symbolic,replacements=positions,
                total_exposures=2700,replaced_exposures=30,other_positions_unchanged=2670,
                negative_positions_unchanged=True,brightness_factors_unchanged=True,
                source_full_class_instance_exposure_preserved=True,
                concentration=dict(Counter(x['source_id'] for x in positions))))
    path=DESIGN/'symbolic-replacement-feasibility.json'
    if path.exists():prior.verify(prior.read(path));return prior.read(path)
    paths=[pp,dp,cp,Path(__file__).resolve()]+[Path(s['source_frame']['source_receipt']) for s in sources.values()]
    return prior.frozen(path,dict(status='symbolic_count_feasible_quality_and_loader_pending',results=results,
        training_ready=False,training_started=False,training_admitted=False,promotable=False,
        limits=['Not exported training sequences: new captures require full-label review and actual label equality.',
                'All four conditions checked for feasibility; no condition selected based on pilot scores.',
                'Warm/cool diversity is replaced by one gray condition; this is a condition strategy, not increased exposure.',
                'Shared sources/assets and S07 concentration retained; no independent-scene claim.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
