"""Identity-checked paired fitting diagnostics, never visual auto-approval."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_routed_backbone_fit import OUT,KEYS
from scripts.vision.routed_backbone_control import SOURCE,SOURCE_KEYS,checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    oldroot=SOURCE.parent/'routed-amplitude-fit-diagnosis-v1';units=[];deps=[OUT/'research-entry.json',Path(__file__)]
    for key,oldkey in zip(KEYS,SOURCE_KEYS):
        ap=OUT/f'{key}.json';bp=oldroot/f'{oldkey}.json'
        a,b=checked(ap),checked(bp);aa={r['member_id']:r for r in a['rows']};bb={r['member_id']:r for r in b['rows']}
        if len(aa)!=144 or len(bb)!=144 or len(a['rows'])!=144 or len(b['rows'])!=144 or aa.keys()!=bb.keys():raise ValueError('Member mismatch')
        transitions=[]
        for mid,r in aa.items():
            prior=bb[mid]
            for f in ('image_sha256','label_sha256','truth','actual_exposures'):
                if r[f]!=prior[f]:raise ValueError('Identity or exposure mismatch '+mid+' '+f)
            oldhits={x['truth_index'] for x in prior['matches']};newhits={x['truth_index'] for x in r['matches']}
            for i,t in enumerate(r['truth']):
                state=('persistent_hit' if i in oldhits else 'gain') if i in newhits else ('loss' if i in oldhits else 'persistent_miss')
                transitions.append(dict(member_id=mid,truth_index=i,truth=t,state=state,image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],actual_exposures=r['actual_exposures']))
        units.append(dict(key=key,old_hits=sum(len(r['matches']) for r in b['rows']),new_hits=sum(len(r['matches']) for r in a['rows']),truth=sum(len(r['truth']) for r in a['rows']),
            misses=dict(Counter(m['reason'] for r in a['rows'] for m in r['misses'])),states=dict(Counter(r['state'] for r in transitions)),transitions=transitions))
        deps += [ap,bp,OUT/f'{key}-verified.json',oldroot/f'{oldkey}-verified.json']
    dest=OUT/'paired-comparison.json'
    if dest.exists():return checked(dest)
    return write_record(dest,dict(status='paired_training_fit_complete_residual_visual_review_pending',units=units,
        inference='Freezing the complete backbone also reduces fitting on identical actually exposed training members in all three seeds; this is not exclusively a development-transfer failure.',
        limits='Supports constrained adaptation, not a unique mechanism or proof of insufficient training steps. Frozen parameters and BN statistics remain confounded. No visual approval, label admission or further training.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in deps}))

if __name__=='__main__':print(run()['status'])
