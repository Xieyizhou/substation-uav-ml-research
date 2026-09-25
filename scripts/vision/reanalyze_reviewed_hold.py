"""Repair in-memory integer mapping keys using persisted protocol; no replay."""
import shutil
import copy
from pathlib import Path
from scripts.vision.reviewed_hold_control import OUT as SOURCE,prior,base,DEPTH,regression

OUT=SOURCE/'normalized-mapping-analysis-v1'


def main():
    pp=SOURCE/'protocol.json';p=prior.read(pp)
    reconstructed=copy.deepcopy(p)
    for f in reconstructed['frames']:
        f['instance_mapping']={int(k):v for k,v in f['instance_mapping'].items()}
    prior.verify(reconstructed)  # Recover the exact pre-serialization integer-key identity.
    results=[];paths=[pp,Path(__file__)]
    OUT.mkdir(exist_ok=True)
    normalized=OUT/'protocol.json'
    if not normalized.exists():
        prior.frozen(normalized,dict(status='integer_key_identity_reconstructed_and_verified',frames=p['frames'],
            training_ready=False,training_started=False,inputs={str(pp):prior.file_sha256(pp),str(Path(__file__)):prior.file_sha256(Path(__file__))}))
    prior.verify(prior.read(normalized));paths.append(normalized)
    old=(base.OUT,base.VARIANT);base.OUT,base.VARIANT=DEPTH,'depth'
    try:
        for f in p['frames']:
            tag=f['review_ids'][0];src=SOURCE/'replay'/tag/'attempt-01';rp=src/'receipt.json';r=prior.read(rp);prior.verify(r)
            if r.get('reason')!='Unknown/duplicate/invalid box' or not r['process_cleanup_complete']:raise ValueError('Not the expected key-type failure')
            folder=OUT/tag;dest=folder/'analysis.json'
            if dest.exists():result=prior.read(dest);prior.verify(result)
            else:
                shutil.copytree(src,folder)
                clock=prior.read(src/'clock-preflight.json');prior.verify(clock)
                fence=base.message_timestamp(clock['samples'][-1]['message'])
                try:result=regression.analyze(folder,f,fence)
                except ValueError as e:result=dict(status='semantic_blocked',reason=str(e))
                result.update(process_cleanup_complete=True,source_receipt=str(rp),replayed=False,
                    inputs={str(x):prior.file_sha256(x) for x in [pp,rp,Path(__file__)]+[x for x in folder.iterdir() if x.is_file()]})
                prior.frozen(dest,result)
            results.append(dict(review_id=tag,member_id=f['member_id'],status=result['status'],reason=result.get('reason'),analysis=str(dest)))
            paths.append(dest);print(tag,result['status'],result.get('reason'),flush=True)
    finally:base.OUT,base.VARIANT=old
    dest=OUT/'completion.json'
    if not dest.exists():prior.frozen(dest,dict(status='all_seven_reanalyzed',results=results,training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':main()
