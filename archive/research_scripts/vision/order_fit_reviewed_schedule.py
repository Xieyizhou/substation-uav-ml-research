"""Versioned minimal-position replacements; brightness and negatives stay put."""
from collections import Counter,defaultdict
import copy
import hashlib
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,TRAIN,prior
from scripts.vision.order_fit_reviewed_counts import DEST,verify_counts
from scripts.vision.freeze_closed_source_control import counts as exposure_counts

VERSION='reviewed-pool-minimal-replacement-v1'


def digest(*parts):return hashlib.sha256('|'.join(map(str,(VERSION,*parts))).encode()).hexdigest()


def replace(old,target,rows,key):
    before=Counter(old);positions=defaultdict(list);holes=defaultdict(list);new=list(old)
    for i,mid in enumerate(old):positions[mid].append(i)
    for mid,positions_for_member in positions.items():
        remove=before[mid]-target.get(mid,0)
        if remove<=0:continue
        if not rows[mid]['class_instances']:raise ValueError('Negative replacement forbidden')
        for i in sorted(positions_for_member,key=lambda i:digest(key,mid,i))[:remove]:
            new[i]=None;holes[rows[mid]['subset']].append(i)
    increments=defaultdict(dict)
    for mid,total in target.items():
        if total>before[mid]:increments[rows[mid]['subset']][mid]=total-before[mid]
    for subset in set(holes)|set(increments):
        remaining=dict(increments[subset]);order=sorted(remaining,key=lambda mid:digest(key,subset,mid,'increase'))
        fillers=[]
        while any(remaining.values()):
            for mid in order:
                if remaining[mid]:fillers.append(mid);remaining[mid]-=1
        if len(fillers)!=len(holes[subset]):raise ValueError('Subset replacement count differs')
        for i,mid in zip(sorted(holes[subset]),fillers):new[i]=mid
    if None in new or dict(Counter(new))!={k:v for k,v in target.items() if v}:raise ValueError('Replacement sequence mismatch')
    for a,b in zip(old,new):
        if rows[a]['subset']!=rows[b]['subset']:raise ValueError('Cross-subset replacement')
        if not rows[a]['class_instances'] and a!=b:raise ValueError('Negative position changed')
    return new


def run():
    dest=DEST/'design.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    pp=OUT/'protocol.json';mp=OUT/'reviewed-dataset-v1/manifest.json';dp=TRAIN/'design.json'
    p,m,old=[prior.read(path) for path in (pp,mp,dp)]
    for r in (p,m,old):prior.verify(r)
    rows={r['member_id']:r for r in p['members']};exported={r['member_id']:r for r in m['members']}
    allowed=set(exported)-set(old['held_members']);paths=[pp,mp,dp,Path(__file__).resolve(),Path(__file__).with_name('order_fit_reviewed_counts.py')]
    result={k:copy.deepcopy(old[k]) for k in ('names','initialization','evaluation','acceptance_policy','environment','training_config','brightness_factors')}
    result.update(pool_rows=list(exported.values()),schedules={},listings={},totals={},ledger={},changes={})
    for key in sorted(old['schedules']):
        cp=DEST/(key+'.json');c=prior.read(cp);prior.verify(c);paths.append(cp)
        if c['status']!='two_stage_optimal_integer_witness_verified':raise ValueError('Counts not solved optimally')
        before=old['schedules'][key];verify_counts(p['members'],Counter(before),c['counts'],allowed)
        seq=replace(before,c['counts'],rows,key)
        if len(seq)!=6600 or any(mid not in allowed for mid in seq):raise ValueError('Bad schedule population')
        result['schedules'][key]=seq
        result['totals'][key]=exposure_counts(seq,exported)
        result['ledger'][key]=[dict(first_step=i//6+1,**exposure_counts(seq[i:i+300],exported)) for i in range(0,6600,300)]
        result['changes'][key]=[dict(position=i,before=a,after=b,brightness=result['brightness_factors'][key][i]) for i,(a,b) in enumerate(zip(before,seq)) if a!=b]
        listing=DEST/(key+'-images.txt');text=''.join(str(Path(exported[mid]['image_path']).resolve())+'\n' for mid in sorted(set(seq)))
        if listing.exists() and listing.read_text()!=text:raise ValueError('Existing listing differs')
        if not listing.exists():listing.write_text(text)
        result['listings'][key]=str(listing);paths.append(listing)
    result.update(status='reviewed_exposure_sequences_frozen_preflight_pending',version=VERSION,
        held_members=sorted(set(rows)-allowed),training_started=False,
        comparison='Reviewed whole-frame holds plus minimum-increase exposure compensation versus each corresponding historical ISR/ISM unit; not a pure label or dose causal intervention.',
        limitations=['Preserves total and subset budgets, full-class instance totals, negative members/positions and position-indexed brightness.',
            'Positive members and some batch compositions change; source concentration and member/brightness combinations are strategy effects.',
            'Previously zero-exposure members remain zero; 376 approved dataset members need not all be sampled in each unit.'])
    return prior.frozen(dest,dict(result,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['status'])
