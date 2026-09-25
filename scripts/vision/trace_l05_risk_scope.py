"""Bounded source/exposure audit and proposal only; no dataset mutation."""
from collections import Counter
from pathlib import Path
import hashlib
from PIL import Image
from scripts.vision.prepare_physical_lighting_control import OUT as PREP, FIT, prior, history
from scripts.vision.run_physical_lighting_capture_v2 import OUT as CAPTURE

OUT=PREP.parent/'l05-supervision-risk-scope-v1'

def pixels(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB')
        return (rgb.size,hashlib.sha256(rgb.tobytes()).hexdigest())

def main():
    pp=PREP/'protocol.json'; fp=FIT/'protocol.json'; ep=history.CONTROL/'evidence.json'
    ledger=PREP.parent/'visual-augmentation-240-v1/intake-ledger.json'
    paths=[pp,fp,ep,ledger,CAPTURE/'blocked-completion.json',CAPTURE/'blocked-evidence/explicit-review.json',Path(__file__).resolve()]
    for path in [pp,fp,ep,*paths[4:6]]: prior.verify(prior.read(path))
    p=prior.read(pp); f=prior.read(fp); source=p['selected'][-1]['source']; mid=source['member']['member_id']
    if p['selected'][-1]['id']!='L05': raise ValueError('Selection changed')
    rec=prior.read(source['source_receipt']); paths += [Path(source['source_receipt']),Path(source['source_image'])]
    entries=prior.read(ledger)['entries']; cid=source['member']['source_candidate_id']
    matches=[x for x in entries if x['candidate_id']==cid]
    if len(matches)!=1: raise ValueError('Source candidate not uniquely registered')
    entry=matches[0]
    siblings=[x for x in entries if x['pose_id']==entry['pose_id'] or x['derivation_group']==entry['derivation_group']]
    registered={x['candidate_id'] for x in siblings}
    targetpixels=pixels(source['source_image']); same=[]; mapped=[]; audit=[]
    for row in f['rows']:
        ip,lp=Path(row['image_path']),Path(row['label_path'])
        if prior.file_sha256(ip)!=row['image_sha256'] or prior.file_sha256(lp)!=row['label_sha256']: raise ValueError('Member input changed')
        paths += [ip,lp]
        classes=Counter(int(line.split()[0]) for line in lp.read_text().splitlines() if line.strip())
        audit.append(dict(member_id=row['member_id'],label_lines=sum(classes.values()),class_ids=dict(classes)))
        if pixels(ip)==targetpixels: same.append(row['member_id'])
        if row['member_id'].removeprefix('candidate:') in registered: mapped.append(row['member_id'])
    affected=sorted(set(same+mapped+[mid])); idx={x['member_id']:x for x in f['rows']}
    exposures={}
    for name,model in f['models'].items():
        draws=model['draws']; counts=Counter(draws)
        exposures[name]=dict(images=sum(counts[x] for x in affected),members={x:counts[x] for x in affected},
            labeled_class_instances=dict(sum((Counter({k:v*counts[x] for k,v in idx[x]['class_instances'].items()}) for x in affected),Counter())))
    slots={seed:dict(affected_replacement_positions=[int(i) for i,m in s['position_to_source'].items() if m in affected],
        remaining_proposed_replacements=sum(m not in affected for m in s['position_to_source'].values()),
        affected_original_exposures=sum(m in affected for m in s['original_sequence'])) for seed,s in p['schedules'].items()}
    OUT.mkdir(exist_ok=True)
    result=prior.frozen(OUT/'scope.json',dict(status='risk_scope_and_counterfactual_only_not_revised_protocol',
        source_member=mid,source_view_id=rec['view_id'],source_family=rec['family'],source_registration=entry,
        registered_same_pose=siblings,pixel_identical_members=same,registered_same_pose_pool_members=mapped,
        proposed_whole_frame_hold=affected,models=exposures,lighting_plan_impact=slots,member_hash_label_audit=audit,
        limitations=['Registered pose relation is not proof that all frames in the same equipment family are identical.',
            'No assertion of complete lineage closure beyond this registered candidate ledger and current 236-member pixel scan.',
            'Holding original exposures changes full-label quotas; skipping only lighting replacement leaves original supervision risk.'],
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print({k:result[k] for k in ['source_view_id','source_family','registered_same_pose_pool_members','pixel_identical_members','models','lighting_plan_impact']})
    print('registered siblings',len(siblings),'pose',entry['pose_id'])

if __name__=='__main__': main()
