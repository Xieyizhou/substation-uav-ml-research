"""Read-only feasibility certificate, label-scale ledger, and saved-world asset audit."""
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.freeze_full_image_training import OUT as PRIOR, REFERENCE, ORIGINAL, ROOT, NAMES, SEEDS, read, save, file_sha256, verify_tree, baseline_verify
from src.vision.canonical.plan import read_record

OUT=PRIOR/'supervision-preservation-feasibility-v1'

def certificate(vectors,target,slots,weights):
    if slots<=0 or not vectors or any(v<0 for row in vectors.values() for v in row.values()):
        raise ValueError('Invalid count constraints')
    score=lambda row:sum(row.get(k,0)*v for k,v in weights.items())
    scores={mid:score(row) for mid,row in vectors.items()};maximum=max(scores.values());demand=score(target)
    saturated=demand==slots*maximum
    return dict(weights=weights,slots=slots,maximum_score_per_image=maximum,required_score=demand,
        saturation_proven=saturated,forced_zero_members=sorted(mid for mid,v in scores.items() if saturated and v<maximum),
        proof='For nonnegative draws x, sum(x)=slots and score<=maximum. At equality every positive draw must attain maximum; any lower-score draw makes the target impossible.')

def labels(row):
    for kind in ('image','label'):
        if file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Stale member bytes')
    with Image.open(row['image_path']) as image:w,h=image.size
    gain=640/max(w,h);result=[];counts=Counter()
    for line in Path(row['label_path']).read_text().splitlines():
        values=list(map(float,line.split()))
        if len(values)!=5 or not all(math.isfinite(x) for x in values):raise ValueError('Malformed YOLO label')
        cls,x,y,bw,bh=values
        if cls!=int(cls) or not 0<=cls<len(NAMES) or not (0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1):
            raise ValueError('Invalid class or normalized dimensions')
        category=NAMES[int(cls)];counts[category]+=1
        result.append(dict(category=category,width_640=bw*w*gain,height_640=bh*h*gain,
                           short_side_640=min(bw*w,bh*h)*gain,area_640=bw*w*bh*h*gain*gain))
    if dict(counts)!=row['class_instances']:raise ValueError('Label supervision metadata mismatch')
    return result

def quantile(pairs,q):
    if not pairs:return None
    total=sum(n for _,n in pairs);position=max(1,math.ceil(q*total));n=0
    for value,count in sorted(pairs):
        n+=count
        if n>=position:return value
    raise ValueError('Invalid weighted quantile')

def scale_ledger(rows,draws,parsed):
    count=Counter(draws);result={}
    for category in NAMES:
        pairs=[(box['short_side_640'],n) for mid,n in count.items() for box in parsed[mid] if box['category']==category]
        result[category]=dict(instance_exposures=sum(n for _,n in pairs),
            short_side_640={f'p{round(q*100)}':quantile(pairs,q) for q in (.1,.5,.9)})
    return result

def model_signature(path,name):
    matches=[m for m in ET.parse(path).findall('.//model') if m.get('name')==name]
    if len(matches)!=1:raise ValueError('Missing or ambiguous model')
    model=matches[0];visuals={}
    for v in model.findall('./link/visual'):
        key=v.get('name')
        if key in visuals:raise ValueError('Duplicate visual component')
        box=v.find('geometry/box/size')
        if box is None:raise ValueError('Unexpected non-box component; inspect separately')
        visuals[key]=dict(shape='box',size=list(map(float,box.text.split())),pose=v.findtext('pose'),
            ambient=v.findtext('material/ambient'),diffuse=v.findtext('material/diffuse'),
            labels=[x.text for x in v.findall('./plugin/label')])
    if set(visuals)!={'body','base','front_panel'}:raise ValueError('Unexpected component topology')
    body=visuals['body']['size'];panel=visuals['front_panel']['size']
    return dict(model_name=name,visuals=visuals,body_aspect=[x/body[0] for x in body],
                panel_body_width_ratio=panel[0]/body[0],panel_body_height_ratio=panel[2]/body[2])

def checked_world(run):
    pp=run/'plan/plan.json';cp=run/'capture/collection-receipt.json';wp=run/'plan/world.sdf'
    plan=read_record(pp);capture=read_record(cp);digest=file_sha256(wp)
    if digest!=plan['files']['world.sdf'] or digest!=capture['world_sha256']:
        raise ValueError('Saved world does not match plan and capture')
    return wp,[pp,cp,wp]

def main():
    dest=OUT/'diagnosis.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING',dest);return
    paths=[PRIOR/'report-receipt.json',PRIOR/'protocol.json',PRIOR/'negative-review.json',REFERENCE/'protocol.json',
           Path(__file__),ROOT/'tests/test_supervision_preservation.py']
    seen=set()
    for path in paths[:4]:verify_tree(path,seen)
    p=read(PRIOR/'protocol.json');old=read(REFERENCE/'protocol.json');members=read(p['dataset_path'])['members']
    lookup={r['member_id']:r for r in old['pool_rows']+members}
    parsed={mid:labels(row) for mid,row in lookup.items()}
    bridge=[r for r in old['pool_rows'] if r['subset']=='bridge_positive']
    vectors={r['member_id']:r['class_instances'] for r in bridge+members};proofs={};scales={}
    for seed in SEEDS:
        seq=old['schedules'][f'I-300-{seed}']
        for block in range(3):
            ids=[m for m in seq[block*600:(block+1)*600] if lookup[m]['subset']=='bridge_positive']
            target=Counter()
            for m in ids:target.update(lookup[m]['class_instances'])
            proof=certificate(vectors,target,len(ids),dict(capacitor_bank=1,reactor=1))
            proof['target_class_instances']=dict(target)
            proof['blocked_new_pairs']=sorted({r['pair_id'] for r in members if r['member_id'] in proof['forced_zero_members']})
            if len(ids)!=120 or not proof['saturation_proven'] or proof['blocked_new_pairs']!=['F03','F08']:
                raise ValueError('Expected infeasibility witness not reproduced')
            proofs[f'{seed}:{block}']=proof
        for arm,schedule in [('I',seq),('K',p['schedules'][f'K-300-{seed}']),('L',p['schedules'][f'L-300-{seed}'])]:
            scales[f'{arm}-300-{seed}']=scale_ledger(lookup,schedule,parsed)
    wp,refs=checked_world(ORIGINAL);paths+=refs
    assets={'positive_entry_switchgear':model_signature(wp,'entry_switchgear')}
    negative_worlds={}
    for d in read(PRIOR/'negative-review.json')['decisions']:
        run=Path(d['image_path']).parent.parent.parent
        if str(run) in negative_worlds:continue
        world,refs=checked_world(run);paths+=refs
        negative_worlds[str(run)]={name:model_signature(world,name) for name in ('cabinet_center','control_building')}
    inputs={str(path):file_sha256(path) for path in paths}
    for row in lookup.values():
        for kind in ('image','label'):inputs[row[kind+'_path']]=row[kind+'_sha256']
    result=subprocess.run([sys.executable,'-m','unittest','tests.test_supervision_preservation','tests.test_full_image_training_freeze','tests.test_matched_appearance_training'],cwd=ROOT,capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    for name in ('test_full_image_training_freeze','test_matched_appearance_training'):
        path=ROOT/'tests'/f'{name}.py';inputs[str(path)]=file_sha256(path)
    OUT.mkdir(parents=True,exist_ok=True)
    save(dest,dict(status='diagnosis_complete_four_class_replacement_infeasible_under_stated_constraints',
        constraints=dict(common_member_positions='fixed_to_I',bridge_slots_per_600=120,
            bridge_class_counts_per_600=dict(transformer=144,switchgear=195,capacitor_bank=42,reactor=78),
            candidate_pool='39 retained bridge images plus 56 new reviewed frames',require_all_new_poses=True),
        infeasibility_certificates=proofs,scale_ledger=scales,asset_signatures=assets,negative_worlds=negative_worlds,
        interpretation=['Certificate applies only to the explicitly fixed common positions, bridge slot budget and available pool; not all possible training designs.',
            'All available images score capacitor+reactor <= 1. Target requires 120 in 120 bridge slots, so new switchgear-only images with score 0 cannot be used.',
            'Shared box/body/base/panel topology and panel material do not imply identical assets, pixel indistinguishability, or wrong labels.',
            '640 short sides use aspect-preserving resize gain; distributions are exposure-weighted, not independent samples.'],
        next_design_status='held_not_frozen_for_training',training_started=False,collection_started=False,
        next_checks=['Inspect retained bridge sources for legally replayable high-cooccurrence positive views with discriminative switchgear panels; do not admit by vector alone.',
            'If adding switchgear-only draws, supply compensating score>1 frames and solve all class and coverage constraints, or explicitly change one experimental constraint.',
            'Audit asset distinctiveness on diagnostic copies before proposing asset revisions; do not relabel ordinary cabinets or buildings as targets.'],
        regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),inputs=inputs))
    print('DIAGNOSIS_COMPLETE_TRAINING_HELD',dest,flush=True)

if __name__=='__main__':main()
