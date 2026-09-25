"""Freeze one whole-sequence budget comparison; no training by default."""
import copy
from pathlib import Path
from scripts.vision.train_closed_source_control import OUT as SOURCE, contract, prior
from scripts.vision.summarize_closed_pool_fit import OUT as FIT, main as fit_summary
from scripts.vision.freeze_closed_source_control import counts

OUT=SOURCE/'repeated-budget-control-v1'
KEYS=tuple(f'B{steps}-{seed}' for steps in (450,900) for seed in (7,17,27))

def check(p,old):
    rows={r['member_id']:r for r in p['pool_rows']}
    for key in KEYS:
        steps=int(key.split('-')[0][1:]);seed=key.split('-')[-1];base='M-'+seed;repeat=steps//450
        if p['schedules'][key]!=old['schedules'][base]*repeat:raise ValueError('Sequence drift')
        if p['brightness_factors'][key]!=old['brightness_factors'][base]*repeat:raise ValueError('Brightness drift')
        expected={**old['training_config'][base],'epochs':steps//10}
        if p['training_config'][key]!=expected:raise ValueError('Additional factor changed')
        if any(m in p['held_members'] for m in p['schedules'][key]):raise ValueError('Held exposure')
        if p['totals'][key]!=counts(p['schedules'][key],rows):raise ValueError('Ledger drift')
    for seed in (7,17,27):
        if p['schedules'][f'B900-{seed}'][:2700]!=p['schedules'][f'B450-{seed}']:raise ValueError('Prefix drift')

def freeze():
    old,_,_=contract('M-7');fit_summary();dest=OUT/'design.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);check(p,old);return p
    p={k:copy.deepcopy(old[k]) for k in ('pool_rows','names','initialization','evaluation','held_members','acceptance_policy','environment')}
    p.update(schedules={},brightness_factors={},training_config={},listings={},totals={},ledger={})
    rows={r['member_id']:r for r in old['pool_rows']}
    for key in KEYS:
        steps=int(key.split('-')[0][1:]);seed=key.split('-')[-1];base='M-'+seed;repeat=steps//450
        p['schedules'][key]=old['schedules'][base]*repeat;p['brightness_factors'][key]=old['brightness_factors'][base]*repeat
        p['training_config'][key]={**old['training_config'][base],'epochs':steps//10};p['listings'][key]=old['listings'][base]
        p['totals'][key]=counts(p['schedules'][key],rows)
        p['ledger'][key]=[dict(first_step=i//6+1,**counts(p['schedules'][key][i:i+300],rows)) for i in range(0,steps*6,300)]
    paths=[SOURCE/'protocol.json',FIT/'summary.json',FIT/'protocol.json',Path(__file__).resolve(),SOURCE/'evaluation/error-review-v1/report-receipt.json']
    p.update(status='design_frozen_runtime_benchmark_and_real_loader_pending',
        comparison='Independent v2.11 initializations: 450 versus 900 steps, exact first-450 sequence; 900 repeats all 2700 exposures and brightness coefficients, not only material examples.',
        scientific_scope='Total budget changes all image/class/negative exposures together. Not a pure material-dose cause, and not proof that insufficient steps is the sole mechanism.',
        cpu_policy='Benchmark training separately before protocol signing; identical effective threads/concurrency policy for both budgets. Fresh 450 control avoids confounding historical unverified thread setup.',
        endpoint_policy='Only prespecified terminal last.pt; no favorable checkpoint or seed selection.',
        stop_policy='Complete all six cells regardless of scores; no extra budget, collection, labels, threshold changes or sealed testing.',
        quality_scope='Reuse only existing M exposure members and their current quality decisions; no new member or previously held member enters. Unknown development boxes are not training approvals.',
        training_started=False,training_admitted=False,promotable=False,inputs={str(x):prior.file_sha256(x) for x in paths})
    OUT.mkdir(exist_ok=True);check(p,old);return prior.frozen(dest,p)

if __name__=='__main__':print(freeze()['status'])
