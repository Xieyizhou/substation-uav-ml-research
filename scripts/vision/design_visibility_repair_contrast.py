"""Auditable prospective schedules; no training pool export or admission."""
import random
from collections import Counter
from pathlib import Path
from scripts.vision.finalize_remaining_bridge_review import OUT as SOURCE,ROOT,REFERENCE,read,save,file_sha256,verify_tree,baseline_verify
from scripts.vision.diagnose_supervision_preservation import labels,scale_ledger

OUT=SOURCE/'visibility-repair-contrast-design-v1'
VARIANTS=('original','neutral_bridge','background_bridge')

def schedule(prior,rows,witness,trace,seed):
    lookup={r['member_id']:r for r in rows};by_id={f['member_id']:f for f in trace['frames']}
    index={(r['lineage_id'],by_id[r['member_id']]['variant']):r['member_id'] for r in rows if r['member_id'] in by_id}
    original=[];mixed=[]
    for block in range(3):
        candidates=[];offset=0
        for w in witness:
            for j in range(w['draws_per_600_total_images']):candidates.append(index[w['lineage_id'],VARIANTS[(j+offset+block)%3]])
            offset+=w['draws_per_600_total_images']
        if len(candidates)!=120:raise ValueError('Bridge budget mismatch')
        random.Random(f'visibility-repair:{seed}:{block}').shuffle(candidates);cursor=0
        for mid in prior[block*600:(block+1)*600]:
            if lookup[mid]['subset']=='bridge_positive':
                chosen=candidates[cursor];cursor+=1
                mixed.append(chosen);original.append(index[lookup[chosen]['lineage_id'],'original'])
            else:mixed.append(mid);original.append(mid)
        if cursor!=120:raise ValueError('Prior slot mismatch')
    return original,mixed

def verify_pair(original,mixed,prior,lookup,parsed,allowed):
    if not len(original)==len(mixed)==len(prior)==1800:raise ValueError('Wrong step budget')
    for o,m,p in zip(original,mixed,prior):
        if lookup[p]['subset']!='bridge_positive' and (o!=p or m!=p):raise ValueError('Common exposure position changed')
        if lookup[p]['subset']=='bridge_positive':
            if o not in allowed or m not in allowed:raise ValueError('Held bridge member used')
            if lookup[o]['lineage_id']!=lookup[m]['lineage_id'] or parsed[o]!=parsed[m]:raise ValueError('Paired geometry/scale mismatch')

def main():
    dest=OUT/'design.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    verify_tree(SOURCE/'completion.json');prior=read(REFERENCE/'protocol.json')
    diagnosis=read(SOURCE/'diagnosis.json');witness=read(SOURCE/'repair-budget-feasibility.json')['witness']
    trace_path=SOURCE.parent/'trace.json';trace=read(trace_path)
    allowed={mid for g in diagnosis['groups'].values() if not g['content_risk_ids'] for mid in g['member_ids']}
    if len(allowed)!=15:raise ValueError('Reviewed scope changed')
    lookup={r['member_id']:r for r in prior['pool_rows']};parsed={mid:labels(row) for mid,row in lookup.items()}
    index={f['member_id']:f for f in trace['frames']};schedules={};ledger={};scales={}
    for seed in (7,17,27):
        old=prior['schedules'][f'I-300-{seed}'];o,m=schedule(old,prior['pool_rows'],witness,trace,seed)
        verify_pair(o,m,old,lookup,parsed,allowed)
        for arm,draws in [('original_only',o),('three_variant',m),('historical_I',old)]:
            key=f'{arm}-300-{seed}';scales[key]=scale_ledger(lookup,draws,parsed)
            if arm=='historical_I':continue
            schedules[key]=draws;blocks=[]
            for b in range(3):
                segment=draws[b*600:(b+1)*600];bridge=[mid for mid in segment if lookup[mid]['subset']=='bridge_positive'];counts=Counter()
                for mid in bridge:counts.update(lookup[mid]['class_instances'])
                if dict(counts)!={'transformer':144,'switchgear':195,'capacitor_bank':42,'reactor':78}:raise ValueError('Class exposure quota mismatch')
                variants=Counter(index[mid]['variant'] for mid in bridge)
                if variants!=(Counter(original=120) if arm=='original_only' else Counter({v:40 for v in VARIANTS})):raise ValueError('Variant quota mismatch')
                blocks.append(dict(subsets=dict(Counter(lookup[mid]['subset'] for mid in segment)),bridge_class_instances=dict(counts),bridge_variants=dict(variants),bridge_lineages=dict(Counter(lookup[mid]['lineage_id'] for mid in bridge))))
            ledger[key]=dict(blocks=blocks,member_exposures=dict(Counter(draws)))
    paths=[SOURCE/'completion.json',SOURCE/'repair-budget-feasibility.json',REFERENCE/'protocol.json',trace_path,Path(__file__),ROOT/'scripts/vision/diagnose_supervision_preservation.py']
    inputs={str(p):file_sha256(p) for p in paths}
    for row in lookup.values():
        for kind in ('image','label'):inputs[row[kind+'_path']]=row[kind+'_sha256']
    save(dest,dict(status='prospective_design_checked_training_preflight_pending',schedules=schedules,exposure_ledger=ledger,scale_ledger=scales,
        allowed_bridge_members=sorted(allowed),group_quota=witness,controls={**prior['controls'],'optimizer_steps':[300],'device':'cpu','warmup_epochs':0},
        acceptance_policy=prior['acceptance_policy'],retention={**prior['retention'],'additional_reference':'matched_original_only_arm'},
        interpretation='Same-slot, same-lineage, same-label geometry contrast: original-only versus existing three-variant bundle. Not a pure material or lighting effect and not a pure quality-repair effect versus historical I.',
        pretraining_requirements=['Revalidate hash-bound input reviews, weights and common-pool admission receipts; this audit newly reviewed only bridge members.',
            'Export a separate development-only dataset and validate exact sampler actual exposure before any optimizer steps.',
            'Bind frozen 48 paired and 48 no-target development evaluation identities, historical A and same-budget R references; do not open sealed tests.',
            'Confirm augmented settings disabled, AdamW constant 0.001, CPU640 batch6, no warmup, terminal300 weights, seeds7/17/27.',
            'Record exact exposure, losses, optimizer steps, independent lineage count and resume checks; no favorable seed selection.'],
        no_new_training_started=True,no_dataset_exported=True,baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),inputs=inputs))
    print('PROSPECTIVE_DESIGN_CHECKED',OUT,flush=True)

if __name__=='__main__':main()
