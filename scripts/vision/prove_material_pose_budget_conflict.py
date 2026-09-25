"""Exact, solver-independent label-count obstruction; never modifies data."""
from collections import Counter
from pathlib import Path
from scripts.vision.merge_material_pose_candidates import OUT,prior
from scripts.vision.train_frozen_multiscale import contract


def witness(old_counts,new_counts):
    score=lambda x:x.get('capacitor_bank',0)+x.get('reactor',0)
    if not old_counts or not new_counts:raise ValueError('Missing vectors')
    if any(score(x)!=1 for x in old_counts.values()):raise ValueError('Old invariant does not hold')
    if any(score(x)>1 for x in new_counts.values()):raise ValueError('Compensating source exists; proof invalid')
    deficit={mid:1-score(x) for mid,x in new_counts.items() if score(x)<1}
    if not deficit:raise ValueError('No strict deficit')
    return dict(deficit_per_exposure=deficit,
        minimum_deficit_all_poses_once=sum(deficit.values()),
        minimum_deficit_all_three_appearances=sum(deficit.values())*3)


def labels(row,names):
    for kind in ('image','label'):
        if prior.file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Stale '+kind)
    counts=Counter()
    for line in Path(row['label_path']).read_text().splitlines():
        parts=line.split()
        if len(parts)!=5:raise ValueError('Malformed full label')
        cls=int(parts[0])
        counts[names[cls]]+=1
    if dict(counts)!=row['class_instances']:raise ValueError('Full label counts mismatch')
    return dict(counts)


def main():
    mp=OUT/'candidate-export-v1/manifest.json';m=prior.read(mp);prior.verify(m)
    if m['reference_overlap_gaps']:raise ValueError('Reference overlap')
    results={};paths=[mp,Path(__file__),prior.ROOT/'docs/material-view-diversity-design-v1.md']
    for seed in (7,17,27):
        p,s,e=contract(f'fixed-{seed}');names=s['names']
        if isinstance(names,dict):names={int(k):v for k,v in names.items()}
        old={x['member_id']:labels(x,names) for x in s['pool_rows'] if x['subset']=='bridge_positive'}
        new={x['member_id']:labels(x,names) for x in m['members'] if x['variant']=='original'}
        proof=witness(old,new)
        pool={x['member_id']:x for x in s['pool_rows']}
        positions=[i for i,mid in enumerate(e['actual']) if pool[mid]['subset']=='bridge_positive']
        if len(positions)!=540:raise ValueError('Bridge budget changed')
        results[str(seed)]=dict(proof,old_vectors=old,new_vectors=new,bridge_positions=len(positions),
            required_capacitor_plus_reactor_exposure=540,
            maximum_with_every_new_pose_once=540-proof['minimum_deficit_all_poses_once'],
            maximum_with_three_exposures_per_pose=540-proof['minimum_deficit_all_three_appearances'])
        paths += [Path(p['source_protocol'])]+[Path(x['label_path']) for x in s['pool_rows'] if x['subset']=='bridge_positive']
    prior.frozen(OUT/'budget-conflict-proof.json',dict(status='original_bridge_only_design_infeasible_for_current_36_candidates',results=results,
        proof='Every old bridge vector has capacitor+reactor=1. Every new vector has <=1, and three mandatory switchgear poses have 0. At fixed 540 bridge slots, any positive exposure to those poses strictly reduces this sum. Rearranging or increasing other old bridge members cannot repair it.',
        training_ready=False,training_started=False,labels_modified=False,
        scope='Current 36 images plus existing bridge members, only bridge replacement, unchanged other subsets and exact class-instance totals. Does not claim every possible future dataset is infeasible.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('PROVED_ALL_SEEDS; DEFICIT >=3 (one per pose) OR >=9 (three per pose); NO_TRAINING')


if __name__=='__main__':main()
