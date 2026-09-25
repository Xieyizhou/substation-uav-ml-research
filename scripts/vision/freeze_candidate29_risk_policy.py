"""Freeze unified hold recommendations and read-only boundary recheck, no solve."""
from pathlib import Path
from scripts.vision.validate_boundary_roundoff import OUT,prior,normalize_box
from scripts.vision.audit_candidate29_completeness import OUT as AUDIT,capture
from scripts.vision.summarize_candidate29_evidence import analyze_raw
from scripts.vision.candidate29_review_pages import raw_boxes
from scripts.vision.prepare_physical_lighting_control import SOURCE,FIT
from scripts.vision.trace_l05_risk_scope import pixels

def main():
    paths=[OUT/'boundary-validation.json',AUDIT/'protocol.json',AUDIT/'replay-index.json',AUDIT/'screening.json',SOURCE/'protocol.json',FIT/'protocol.json',Path(__file__).resolve()]
    for x in paths[:-1]:prior.verify(prior.read(x))
    p=prior.read(paths[1]);index=prior.read(paths[2]);training=prior.read(paths[4]);pool=prior.read(paths[5])['rows']
    byid={x['pair_id']:x for x in p['frames']};rechecks=[];adjustments=[]
    original=capture.base.box_map
    def diagnostic_boxes(message,mapping):
        raw=raw_boxes(message);result={}
        for label,box in raw.items():
            if label not in mapping:raise ValueError('Unresolved instance label')
            normalized=normalize_box(box);result[label]=normalized['normalized']
            if normalized['changes']:adjustments.append(dict(label=label,**normalized))
        return result
    try:
        capture.base.box_map=diagnostic_boxes
        for item in index['results']:
            frame=byid[item['pair_id']];rp=Path(item['receipt']);prior.verify(prior.read(rp));paths.append(rp)
            records,rawpaths=analyze_raw(frame,rp.parent);paths+=rawpaths;r=records[0]
            rechecks.append(dict(pair_id=frame['pair_id'],member_id=frame['member_id'],records=records,
                membership_conflict=bool(r['added'] or r['lost'] or r['unboxed'] or any(x>1 for x in r['deltas'].values()))))
    finally:capture.base.box_map=original
    if [r['pair_id'] for r in rechecks if r['membership_conflict']]!=['A25','A28']:raise ValueError('Unexpected risk change')
    ledger=AUDIT.parent/'visual-augmentation-240-v1/intake-ledger.json';entries=prior.read(ledger)['entries'];paths.append(ledger)
    scope=[];newholds=set()
    for pid in ('A25','A28'):
        frame=byid[pid];mid=frame['member_id'];cid=mid.removeprefix('candidate:');matches=[x for x in entries if x['candidate_id']==cid]
        if len(matches)!=1:raise ValueError('Risk source not unique')
        entry=matches[0];siblings=[x for x in entries if x['pose_id']==entry['pose_id'] or x['derivation_group']==entry['derivation_group']]
        registered={'candidate:'+x['candidate_id'] for x in siblings};target=pixels(frame['source_image']);same=[]
        for row in pool:
            for kind in ('image','label'):
                path=Path(row[kind+'_path'])
                if prior.file_sha256(path)!=row[kind+'_sha256']:raise ValueError('Changed pool')
                paths.append(path)
            if pixels(row['image_path'])==target:same.append(row['member_id'])
        affected=sorted(({row['member_id'] for row in pool}&registered)|set(same)|{mid});newholds.update(affected)
        scope.append(dict(pair_id=pid,member_id=mid,source_registration=entry,registered_same_pose=siblings,pixel_identical_pool_members=same,affected_members=affected))
    oldheld=set(training['held_members']);held=oldheld|newholds
    prior.frozen(OUT/'policy.json',dict(status='unified_risk_policy_frozen_diagnostic_recheck_complete',
        original_held_members=sorted(oldheld),new_held_members=sorted(newholds),future_zero_exposure_members=sorted(held),source_scope=scope,
        diagnostic_rechecks=rechecks,boundary_adjustment_log=adjustments,
        boundary_policy='Diagnostic-only one-float32-ULP endpoint normalization with original coordinates retained. Existing capture gate and historical labels unchanged; not a deployment or training-admission change.',
        constraints=['No legacy hold or zero exposure restored','No solving or new sampling sequence in this stage','Future same-source confirmed variants share hold role','Full-frame screening is not per-label quality approval'],
        next_gate='Explicit full-label content approvals for future increment members; a separately authorized single new solve for paired revised reference/light treatment only after quality readiness.',
        training_ready=False,training_started=False,dataset_exported=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('POLICY_FROZEN',len(oldheld),'OLD_HOLDS',len(newholds),'NEW_HOLDS',len(held),'TOTAL');print([(x['pair_id'],x['affected_members']) for x in scope])

if __name__=='__main__':main()
