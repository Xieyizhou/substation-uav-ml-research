"""Freeze source-level late-exposure feasibility; does not start training."""
from collections import Counter
from pathlib import Path
from scripts.vision.freeze_material_retention_coverage import freeze as original,OUT as SOURCE,prior
from scripts.vision.material_late_rehearsal_solver import solve,VERSION

OUT=SOURCE.parent/VERSION

def main():
    p=original();rows={r['member_id']:r for r in p['pool_rows']}
    analysis=SOURCE.parent/'material-learning-trajectory-v1/analysis-v1/completion-v2.json'
    prior.verify(prior.read(analysis))
    cells={}
    for seed in (7,17,27):
        key=f'T-{seed}';seq=p['schedules'][key]
        slots=[i for i,m in enumerate(seq) if 'full_truth' in rows[m] and rows[m].get('variant') in ('warm','cool','gray_target_body')]
        if len(slots)!=30:raise ValueError('Material slot budget drift')
        source={seq[i]:rows[seq[i]]['pair_id'] for i in slots}
        solved=solve([seq[i] for i in slots],source,seed)
        new=list(seq)
        for i,m in zip(slots,solved['members'],strict=True):new[i]=m
        if Counter(new)!=Counter(seq):raise ValueError('Member budget changed')
        if any(new[i]!=seq[i] for i in range(2700) if i not in slots):raise ValueError('Non-material slot changed')
        ledger=[]
        for start in range(0,2700,300):
            part=new[start:start+300]
            ledger.append(dict(first_step=start//6+1,members=dict(Counter(part)),
                class_instances=dict(sum((Counter(rows[m]['class_instances']) for m in part),Counter())),
                lineages=dict(Counter(rows[m]['lineage_id'] for m in part))))
        cells[f'L-{seed}']=dict(reference=key,seed=seed,sequence=new,material_positions=slots,solution=solved,
            earliest_last_source_step=slots[solved['optimal_minimum_last_exposure_slot']]//6+1,
            brightness_factors=p['brightness_factors'][key],training_config=p['training_config'][key],ledger=ledger)
    paths=[SOURCE/'protocol.json',analysis,Path(__file__).resolve(),Path(__file__).with_name('material_late_rehearsal_solver.py')]
    OUT.mkdir(exist_ok=True);dest=OUT/'feasibility.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r)
        if r['cells']!=cells:raise ValueError('Non-reproducible sequence')
        return r
    return prior.frozen(dest,dict(status='feasible_sequences_frozen_real_loader_preflight_pending',cells=cells,
        source_protocol_identity=p['identity'],training_ready=False,training_started=False,
        selection_rule='All registered sources, independent of model error membership.',
        limits=['Batch composition and member/brightness associations can change.',
                'Same counts do not certify optimal learning or sufficient independent coverage.',
                'Max-min property is over image slots; two slots may share an optimization step.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    r=main();print(r['status'])
    for k,c in r['cells'].items():print(k,c['solution']['minimum_changed_slots'],'changed slots; all sources last seen at/after',c['earliest_last_source_step'])
