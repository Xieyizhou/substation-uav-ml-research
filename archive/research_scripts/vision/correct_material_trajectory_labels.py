"""Versioned correction of recovered-hit labels; preserve all historical predictions."""
from pathlib import Path
from scripts.vision.analyze_material_learning_trajectory import OUT,prior


def classify(hits):
    if not hits:raise ValueError('Empty trajectory')
    if not hits[-1]:return 'previously_hit_endpoint_miss' if any(hits[:-1]) else 'never_hit_at_observed_steps'
    return 'endpoint_hit_with_intermediate_loss' if any(a and not b for a,b in zip(hits,hits[1:])) else 'endpoint_hit'


def main():
    path=OUT/'summary.json';r=prior.read(path);prior.verify(r)
    from copy import deepcopy
    body=deepcopy({k:v for k,v in r.items() if k not in ('identity','inputs')});changes=[]
    for t in body['trajectories']:
        corrected=classify([x['hit'] for x in t['timeline']])
        if corrected!=t['classification']:
            if not t['endpoint_hit']:raise ValueError('Unexpected endpoint-miss change')
            changes.append(dict(cell=t['cell'],mode=t['mode'],member_id=t['member_id'],truth=t['truth'],
                                old=t['classification'],new=corrected))
            t['classification']=corrected
    body.update(erratum=dict(reason='Pre-learning misses are not an intervening loss; require an adjacent observed hit-to-miss transition.',
                             changes=changes,endpoint_miss_counts_unchanged=True,predictions_unchanged=True),
                inputs={str(path):prior.file_sha256(path),str(Path(__file__).resolve()):prior.file_sha256(__file__)})
    return prior.frozen(OUT/'summary-v2.json',body)


if __name__=='__main__':print('corrected trajectory labels',len(main()['erratum']['changes']))
