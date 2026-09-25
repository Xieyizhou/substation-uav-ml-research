"""Read-only historical exposure impact; no schedule or dataset modification."""
from collections import Counter
from pathlib import Path
from scripts.vision.brightness_lr_retention import OUT as TRAIN
from scripts.vision.freeze_annotation_revision_design import OUT as DESIGN,ROOT,read,verify,frozen,file_sha256
from scripts.vision.test_body_material_applicability import baseline_verify

OUT=DESIGN/'actual-exposure-impact-v1'


def impact(draws,planned,members):
    if len(draws)!=2700 or draws!=planned:raise ValueError('Actual sequence differs from frozen plan')
    counts=Counter(draws);total=Counter();rows=[]
    for r in members:
        n=counts[r['member_id']];loss={k:n*v for k,v in r['class_instances'].items()}
        total.update(loss)
        rows.append(dict(member_id=r['member_id'],actual_image_exposures=n,
            existing_label_instance_exposures=loss,
            positions_zero_based=[i for i,m in enumerate(draws) if m==r['member_id']],
            windows_50_steps=[draws[i:i+300].count(r['member_id']) for i in range(0,2700,300)]))
    return dict(members=rows,whole_frame_hold_counterfactual=dict(image_exposures=sum(r['actual_image_exposures'] for r in rows),
        existing_label_instance_exposures=dict(total),applied=False))


def main():
    dest=OUT/'audit.json'
    if dest.exists():verify(read(dest));print('VALID_AUDIT_REUSED');return
    dp=DESIGN/'design-receipt.json';pp=TRAIN/'protocol.json';sp=DESIGN/'source-review-v1/sibling-replay-v1/completion.json'
    paths=[dp,pp,sp,Path(__file__)]
    for x in paths[:3]:verify(read(x))
    p=read(pp);members=[r for g in read(dp)['affected_source_groups'] for r in g['members']]
    units=[]
    for family in ('brightness_lr0005','brightness_lr001'):
        for seed in (7,17,27):
            key=f'brightness-450-{seed}'
            cp=TRAIN/'training'/key/'completion.json' if family=='brightness_lr0005' else Path(p['baselines'][str(seed)]['training_receipt'])
            c=read(cp);verify(c);ep=Path(c['exposure_path']);e=read(ep);verify(e)
            if c['status']!='complete' or c['optimizer_steps']!=450 or not c['exposure_verified']:raise ValueError('Incomplete training receipt')
            wp=Path(c['weights'])
            if file_sha256(wp)!=c['weights_sha256']:raise ValueError('Changed endpoint weight')
            bp=cp.parent/'brightness-receipt.json';b=read(bp);verify(b)
            units.append(dict(family=family,seed=seed,training_receipt=str(cp),exposure_receipt=str(ep),weights_sha256=c['weights_sha256'],
                **impact(e['draws'],p['schedules'][key],members)))
            paths += [cp,ep,wp,bp]
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    OUT.mkdir(parents=True,exist_ok=True)
    frozen(dest,dict(status='six_historical_units_actual_exposure_verified',units=units,baseline=baseline,
        scope='Latest brightness LR experiment and its three frozen brightness controls only; not all historical models.',
        interpretation='Counterfactual losses describe removal without replacement, not a proposed changed schedule or causal model attribution.',
        training_ready=False,training_started=False,historical_labels_changed=False,data_removed=False,
        inputs={str(x):file_sha256(x) for x in paths}))
    for u in units:print(u['family'],u['seed'],[r['actual_image_exposures'] for r in u['members']],u['whole_frame_hold_counterfactual'])


if __name__=='__main__':main()
