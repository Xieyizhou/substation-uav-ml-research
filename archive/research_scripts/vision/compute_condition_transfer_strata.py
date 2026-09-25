"""Extend fixed geometry/visibility strata across all four paired visual conditions."""
import re
from collections import defaultdict
from pathlib import Path
from scripts.vision.evaluate_unified_lighting import OUT as RUN, prior, validate_record
from scripts.vision.prepare_development_content_strata import OUT

EVAL = RUN / 'evaluation'
CELLS = ['R-clean-7','R-clean-17','R-clean-27','L-physical-7','L-physical-17','L-physical-27']
VARIANTS = ('original','material','background','lighting')

def iid(t): return int(re.search(r'-instance-(\d+)-', t['annotation_id']).group(1))

def main():
    review = prior.read(OUT/'review-v2.json'); prior.verify(review)
    # Only original/lighting decisions are directly reviewed; transfer to other variants is geometry-only,
    # and is allowed only after exact pair/truth identity checks.
    strata = {(x['pair_id'], x['instance_id']): x for x in review['decisions'] if x['variant']=='original'}
    if len(strata) != 60: raise ValueError('Expected 60 original strata decisions')
    inputs=[OUT/'review-v2.json',Path(__file__).resolve()]
    records={}; metrics=defaultdict(lambda: dict(truth=0,hit=0,misses=0,low_confidence=0,wrong_class=0,localization=0,no_retained=0,matching_conflict=0))
    truth_index={}
    for cell in CELLS:
        p=EVAL/f'{cell}.json'; r=prior.read(p); validate_record(r,cell); inputs.append(p); records[cell]=r
        grouped=defaultdict(dict)
        for row in r['rows']:
            if row['variant'] not in VARIANTS: continue
            grouped[row['pair_id']][row['variant']]=row
        if len(grouped)!=12: raise ValueError('Expected 12 complete condition pairs')
        for pair, rows in grouped.items():
            if set(rows)!=set(VARIANTS): raise ValueError('Condition pair incomplete')
            ref={iid(t):(t['class_name'],t['bbox_xyxy']) for t in rows['original']['truth']}
            for variant,row in rows.items():
                cur={iid(t):(t['class_name'],t['bbox_xyxy']) for t in row['truth']}
                if cur!=ref: raise ValueError('Geometry/truth differs across visual conditions')
                for idx,t in enumerate(row['truth']):
                    key=(pair,iid(t)); s=strata.get(key)
                    if s is None: raise ValueError(f'Missing source stratum {key}')
                    m=metrics[(cell,variant,s['stratum'],t['class_name'])];m['truth']+=1
                    if any(x['truth_index']==idx for x in row['matches']): m['hit']+=1
                    else:
                        m['misses']+=1; miss=next(x for x in row['misses'] if x['truth_index']==idx)
                        bucket={'low_confidence_same_class':'low_confidence','wrong_class':'wrong_class','localization':'localization','no_qualifying_retained_prediction':'no_retained'}[miss['reason']]
                        m[bucket]+=1
                    if row['matching_conflict']:m['matching_conflict']+=1
    rows=[]
    for (cell,variant,stratum,category),m in sorted(metrics.items()):
        rows.append(dict(cell=cell,variant=variant,stratum=stratum,category=category,**m,recall=m['hit']/m['truth']))
    prior.frozen(OUT/'condition-transfer-metrics.json',dict(status='condition_transfer_strata_complete',cells=CELLS,variants=VARIANTS,
        propagation='Original geometry/visibility strata were propagated by exact pair_id, instance_id, class and bbox identity; no new visual content decision was inferred.',
        rows=rows,inputs={str(x):prior.file_sha256(x) for x in inputs}))
    print('CONDITION_TRANSFER_STRATA_COMPLETE',len(rows))

if __name__=='__main__': main()
