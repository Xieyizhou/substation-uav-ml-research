"""Join fixed predictions to prediction-free strata; preserve full-set metrics."""
import json
import re
from collections import defaultdict
from pathlib import Path
from scripts.vision.evaluate_unified_lighting import OUT as RUN, prior, validate_record
from scripts.vision.prepare_development_content_strata import OUT

EVAL = RUN / 'evaluation'
CELLS = ['R-clean-7','R-clean-17','R-clean-27','L-physical-7','L-physical-17','L-physical-27']

def iid(t):
    return int(re.search(r'-instance-(\d+)-', t['annotation_id']).group(1))

def main():
    review = prior.read(OUT/'review-v2.json'); prior.verify(review)
    if review['status'] != 'content_strata_review_complete' or len(review['decisions']) != 120: raise ValueError('Review incomplete')
    strata = {(x['view_id'], x['variant'], x['instance_id']): x for x in review['decisions']}
    if len(strata) != 120: raise ValueError('Duplicate strata identity')
    paths = [OUT/'review.json', Path(__file__).resolve()]
    cells = {}
    metrics = defaultdict(lambda: dict(truth=0, hit=0, misses=0, low_confidence=0, wrong_class=0, localization=0, no_retained=0, matching_conflict=0))
    for cell in CELLS:
        path = EVAL / f'{cell}.json'; r = prior.read(path); validate_record(r, cell); paths.append(path); cells[cell] = r
        for row in r['rows']:
            if row['variant'] not in ('original', 'lighting'): continue
            for idx, truth in enumerate(row['truth']):
                key = (row['view_id'], row['variant'], iid(truth))
                d = strata.get(key)
                if d is None: raise ValueError(f'Missing stratum for {key}')
                m = metrics[(cell, row['variant'], d['stratum'], truth['class_name'])]
                m['truth'] += 1
                hit = any(x['truth_index'] == idx for x in row['matches'])
                if hit: m['hit'] += 1
                else:
                    m['misses'] += 1
                    miss = next(x for x in row['misses'] if x['truth_index'] == idx)
                    reason = miss['reason']
                    m[{'low_confidence_same_class':'low_confidence','wrong_class':'wrong_class','localization':'localization','no_qualifying_retained_prediction':'no_retained'}[reason]] += 1
                if row['matching_conflict']: m['matching_conflict'] += 1
    rows=[]
    for (cell, variant, stratum, category), m in sorted(metrics.items()):
        rows.append(dict(cell=cell, variant=variant, stratum=stratum, category=category, **m,
                         recall=m['hit']/m['truth'] if m['truth'] else None,
                         miss_rate=m['misses']/m['truth'] if m['truth'] else None))
    # Full metrics are copied from the already-validated evaluation summary; no filtered score is substituted.
    full = {cell: cells[cell]['summary'] for cell in CELLS}
    prior.frozen(OUT/'metrics-v2.json', dict(status='stratified_metrics_complete', cells=CELLS,
        strata=['clear','partial','fragment','unknown'], rows=rows, full_set_metrics=full,
        interpretation='Stratified metrics are diagnostic overlays. Full-set metrics remain the formal fixed evaluation; no target was excluded.',
        matching_conflicts=sum(x['matching_conflict'] for x in rows),
        inputs={str(x): prior.file_sha256(x) for x in paths}))
    print('STRATIFIED_METRICS_COMPLETE',len(rows))

if __name__ == '__main__': main()
