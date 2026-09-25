"""Current condition coverage and error census; no rendering, inference or training."""
import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
from scripts.vision.brightness_lr_retention import OUT as SOURCE, ROOT, KEYS, ready, complete, read, verify, frozen, file_sha256, baseline_verify
from scripts.vision.build_switchgear_condition_review import OUT as OLD
from scripts.vision.record_switchgear_condition_review import validate
from scripts.vision.audit_structure_exposure import pixels,label_counts
from scripts.vision.structure_fit import truth_for

OUT=SOURCE/'condition-transfer-design-v1'
FOCUS=('switchgear','capacitor_bank')


def unique(rows,key):
    result={}
    for r in rows:
        k=key(r)
        if k in result:raise ValueError('Duplicate identity')
        result[k]=r
    return result


def labels(path):
    return tuple(tuple(map(float,line.split())) for line in Path(path).read_text().splitlines() if line.strip())


@lru_cache(maxsize=None)
def same_source(image,label,old_image,old_label):
    # Exact native RGB and full numeric labels; no approximate image/coordinate match.
    return pixels(image)==pixels(old_image) and labels(label)==labels(old_label)


def summarize_targets(targets):
    result={}
    for cls in FOCUS:
        rows=[x for x in targets if x['truth']['class_name']==cls]
        panels={}
        for condition in sorted({x['panel_condition'] for x in rows}):
            ds=[x for x in rows if x['panel_condition']==condition]
            panels[condition]=dict(frame_label_events=len(ds),members=len({x['member_id'] for x in ds}),
                registered_lineages=len({x['lineage_id'] for x in ds}),
                instance_exposures={str(s):sum(x['actual_exposures'][str(s)] for x in ds) for s in (7,17,27)})
        result[cls]=dict(frame_label_events=len(rows),panel=panels,
            linkage= dict(Counter(x['linkage_status'] for x in rows)))
    return result


def development(records):
    # Compare stable annotation identities AND coordinates, not array position alone.
    indexed=[unique(r['rows'],lambda x:(x['view_id'],x['variant'])) for r in records]
    if any(set(x)!=set(indexed[0]) for x in indexed):raise ValueError('Development membership drift')
    targets=[]
    for key,row in indexed[0].items():
        for other in indexed[1:]:
            if any(row[k]!=other[key][k] for k in ('truth','image_sha256','pair_id')):raise ValueError('Development truth/identity drift')
        unique(row['truth'],lambda t:t['annotation_id'])
        for i,t in enumerate(row['truth']):
            if t['class_name'] not in FOCUS:continue
            states=[]
            for seed,idx in zip((7,17,27),indexed):
                r=idx[key];hit=any(m['truth_index']==i for m in r['matches'])
                miss=next((m for m in r['misses'] if m['truth_index']==i),None)
                if hit==(miss is not None):raise ValueError('Missing/ambiguous match outcome')
                states.append(dict(seed=seed,hit=hit,miss=miss))
            targets.append(dict(view_id=key[0],variant=key[1],pair_id=row['pair_id'],image_sha256=row['image_sha256'],truth=t,states=states,
                review_status='condition_review_pending_not_inferred_from_miss_reason'))
    counts={}
    for v in ('original','material','background','lighting'):
        counts[v]={}
        for cls in FOCUS:
            rows=[r for r in targets if r['variant']==v and r['truth']['class_name']==cls]
            counts[v][cls]=dict(unique_frame_instances=len(rows),persistent_misses=sum(all(not s['hit'] for s in r['states']) for r in rows),
                hits_by_seed={str(s):sum(r['states'][j]['hit'] for r in rows) for j,s in enumerate((7,17,27))},
                miss_event_reasons=dict(Counter(s['miss']['reason'] for r in rows for s in r['states'] if not s['hit'])))
    return targets,counts


def run():
    p=ready();verify(read(SOURCE/'completion.json'));OUT.mkdir(exist_ok=True)
    dest=OUT/'coverage.json'
    if dest.exists():verify(read(dest));return read(dest)
    manifest=read(OLD/'manifest.json');review=read(OLD/'review.json');verify(manifest);verify(review);verify(read(OLD/'completion.json'))
    validate(manifest,review['decisions'])
    oldrows=unique(manifest['items'],lambda x:x['review_id'])
    decisions=unique(review['decisions'],lambda x:(x['member_id'],x['label_line_index']))
    members=unique(p['pool_rows'],lambda x:x['member_id']);paths=[SOURCE/'protocol.json',SOURCE/'completion.json',OLD/'manifest.json',OLD/'review.json',OLD/'completion.json',Path(__file__),ROOT/'scripts/vision/record_switchgear_condition_review.py',ROOT/'scripts/vision/audit_structure_exposure.py',ROOT/'scripts/vision/structure_fit.py']
    counts={};records=[]
    for key in KEYS:
        complete(key,p);c=read(SOURCE/'training'/key/'completion.json');e=read(c['exposure_path']);verify(e)
        counts[key.split('-')[-1]]=Counter(e['draws']);ep=SOURCE/'evaluation'/f'{key}.json';r=read(ep);verify(r)
        if r['matching_conflicts']:raise ValueError('Unresolved matching conflict')
        records.append(r);paths += [Path(c['exposure_path']),ep,SOURCE/'training'/key/'brightness-receipt.json']
    targets=[]
    for r in members.values():
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale member')
            paths.append(Path(r[kind+'_path']))
        if label_counts(r['label_path'],p['names'])!=r['class_instances']:raise ValueError('Full label counts differ')
        for t in truth_for(r):
            if t['class_name'] not in FOCUS:continue
            d=decisions.get((r['member_id'],t['label_line_index']));status='no_linked_condition_review';panel='unknown';occlusion='unknown';source_id=None
            if d is not None:
                old=oldrows[d['review_id']];paths += [Path(old['image_path']),Path(old['label_path'])]
                if same_source(r['image_path'],r['label_path'],old['image_path'],old['label_path']):
                    status='historical_condition_linked_not_new_admission';panel=d['panel_condition'];occlusion=d['occlusion_condition'];source_id=d['review_id']
                else:status='source_or_full_label_changed_requires_new_review'
            targets.append(dict(member_id=r['member_id'],subset=r['subset'],lineage_id=r['lineage_id'],lineage_resolution=r['lineage_resolution'],
                truth=t,image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],linkage_status=status,prior_review_id=source_id,
                panel_condition=panel,occlusion_condition=occlusion,actual_exposures={s:c[r['member_id']] for s,c in counts.items()},
                independent_scene_count='unknown',pixel_visibility_certified=False))
    dev,ds=development(records)
    old_low=[d for d in review['decisions'] if d['panel_condition']=='visible_low_contrast']
    removed=[d['review_id'] for d in old_low if d['member_id'] not in members]
    return frozen(dest,dict(status='coverage_census_complete_condition_review_required',training_started=False,training_ready=False,
        current_pool_members=len(members),targets=targets,summary=summarize_targets(targets),development_targets=dev,development_summary=ds,
        old_low_contrast_reviews_outside_current_pool=removed,
        interpretation='Historical conditions are reused only for identical native RGB and complete labels at the same member/line. Unknown is retained; this is not new visual review or data admission.',
        inputs={str(x):file_sha256(x) for x in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-census',action='store_true');a=ap.parse_args()
    if a.freeze_census:r=run();print(r['status']);print(r['summary']);print(r['development_summary'])
    else:print('READ_ONLY_NO_RENDERING_NO_INFERENCE_NO_TRAINING')
