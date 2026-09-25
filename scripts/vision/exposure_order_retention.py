"""Frozen error coverage and batch-order-only development experiment."""
import argparse
import random
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.prepare_retention450_training import OUT as PRIOR,ROOT,read,save,file_sha256,verify_tree,baseline_verify
from scripts.vision.exposure_protocol import exposures,NAMES,verify
from src.ml.artifacts import object_sha256

OUT=PRIOR/'exposure-order-retention-v1'
KEYS=tuple(f'{arm}-450-{seed}' for seed in (7,17,27) for arm in ('staged','interleaved'))

def frozen(path,record):
    if path.exists():
        old=read(path);verify(old)
        raise ValueError(f'Preserve frozen artifact: {path}')
    return save(path,record)

def identity(t):
    # Sensor annotation embeds the stable simulator instance label.
    import re
    found=re.search(r'-instance-(\d+)-',t['annotation_id'])
    if not found:raise ValueError('Unresolved simulator instance identity')
    return found.group(1),t['class_name'],tuple(t['bbox_xyxy'])

def panel(ip,box,path,title,predictions=()):
    im=Image.open(ip).convert('RGB');full=im.copy();d=ImageDraw.Draw(full)
    for pred in predictions:d.rectangle(pred['bbox_xyxy'],outline='red',width=3)
    d.rectangle(box,outline='lime',width=5)
    page=Image.new('RGB',(1200,540),'white');full.thumbnail((800,450));page.paste(full,(0,65))
    crop=im.crop(tuple(box));crop.thumbnail((390,450));page.paste(crop,(805,65))
    ImageDraw.Draw(page).text((8,10),title,fill='black');page.save(path)

def evidence():
    dest=OUT/'evidence.json'
    if dest.exists():verify_tree(dest);return read(dest)
    OUT.mkdir(parents=True,exist_ok=True);p=read(PRIOR/'protocol.json')
    paths=[PRIOR/'numerical-audit-v1.json',PRIOR/'protocol.json',Path(__file__)]
    maps={}
    for kind in ('negative','paired'):
        rp=Path(p['evaluation'][kind+'_review']);paths.append(rp)
        maps[kind]={(r['view_id'],r['variant']):r for r in read(rp)['frames']}
    inputs={str(x):file_sha256(x) for x in paths};negative=[];reactors={};records={}
    for seed in (7,17,27):
        for arm in ('retained_reference','retained_appearance'):
            key=f'{arm}-450-{seed}';ep=PRIOR/f'evaluation-{key}.json'
            r=read(ep);verify(r);records[key]=r;inputs[str(ep)]=file_sha256(ep)
            if r['status']!='complete' or r['matching_conflicts']:raise ValueError('Invalid evaluation')
            for row in r['negative_rows']:
                src=maps['negative'][row['view_id'],row['variant']];ip=Path(src['image_path'])
                if file_sha256(ip)!=row['image_sha256']:raise ValueError('Stale negative image')
                for i,box in enumerate(row['predictions']):
                    eid=f'N{len(negative)+1:02}';path=OUT/f'{eid}.png'
                    panel(ip,box['bbox_xyxy'],path,f'{eid} {key} {row["variant"]} {box["class_name"]} {box["confidence"]:.4f}')
                    negative.append(dict(event_id=eid,cell=key,seed=seed,view_id=row['view_id'],variant=row['variant'],
                        image_path=str(ip),image_sha256=row['image_sha256'],prediction_index=i,prediction=box,
                        evidence_path=str(path),evidence_sha256=file_sha256(path)))
                    inputs[str(ip)]=file_sha256(ip);inputs[str(path)]=file_sha256(path)
        a=records[f'retained_reference-450-{seed}'];b=records[f'retained_appearance-450-{seed}']
        amap={(x['pair_id'],x['variant']):x for x in a['rows']}
        for row in b['rows']:
            if row['variant'] not in ('original','lighting'):continue
            old=amap[row['pair_id'],row['variant']]
            a_ids={identity(t):i for i,t in enumerate(old['truth'])};b_ids={identity(t):i for i,t in enumerate(row['truth'])}
            if len(a_ids)!=len(old['truth']) or len(b_ids)!=len(row['truth']) or set(a_ids)!=set(b_ids):raise ValueError('Instance collision / coordinates mismatch')
            for ident,i in b_ids.items():
                if ident[1]!='reactor':continue
                j=a_ids[ident];key=(row['pair_id'],row['variant'],ident)
                if key not in reactors:
                    src=maps['paired'][row['view_id'],row['variant']];ip=Path(src['image_path'])
                    if file_sha256(ip)!=row['image_sha256']:raise ValueError('Stale reactor image')
                    eid=f'R{len(reactors)+1:02}';path=OUT/f'{eid}.png';box=row['truth'][i]['bbox_xyxy']
                    panel(ip,box,path,f'{eid} {row["variant"]} reactor instance {ident[0]}')
                    reactors[key]=dict(event_id=eid,pair_id=row['pair_id'],view_id=row['view_id'],variant=row['variant'],instance_label=ident[0],
                        bbox_xyxy=box,short_side_at_640=min(box[2]-box[0],box[3]-box[1])/3,
                        image_path=str(ip),image_sha256=row['image_sha256'],evidence_path=str(path),evidence_sha256=file_sha256(path),comparisons=[])
                    inputs[str(ip)]=file_sha256(ip);inputs[str(path)]=file_sha256(path)
                comparison={'seed':seed,'planned':i==row['planned_truth_index']}
                for role,rr,idx in [('reference',old,j),('appearance',row,i)]:
                    comparison[role]=dict(hit=any(m['truth_index']==idx for m in rr['matches']),
                        misses=[m for m in rr['misses'] if m['truth_index']==idx],
                        formal_predictions=rr['predictions'],low_predictions=rr['low_predictions'])
                reactors[key]['comparisons'].append(comparison)
    if len(negative)!=24 or len({x['image_sha256'] for x in negative})!=12:raise ValueError('Negative coverage changed')
    if len(reactors)!=10 or sum(len(r['comparisons']) for r in reactors.values())!=30:raise ValueError('Reactor coverage incomplete')
    return frozen(dest,dict(status='pending_explicit_review',negative=negative,reactors=list(reactors.values()),inputs=inputs))

def permutation(sequence,variant_by_mid,seed):
    if len(sequence)!=2700:raise ValueError('Expected 450 complete batches')
    batches=[sequence[i:i+6] for i in range(0,len(sequence),6)]
    appearance=[];other=[]
    for i,batch in enumerate(batches):
        if any(m not in variant_by_mid for m in batch):raise ValueError('Unresolved member role')
        if any(variant_by_mid[m] in ('neutral_bridge','background_bridge') for m in batch):appearance.append(i)
        else:other.append(i)
    if not appearance or not other:raise ValueError('Both batch types required')
    random.Random(f'exposure-order-retention-v1:{seed}:appearance').shuffle(appearance)
    random.Random(f'exposure-order-retention-v1:{seed}:other').shuffle(other)
    ids=[];ai=oi=0;n=len(appearance)
    for i in range(450):
        if ((i+1)*n)//450>(i*n)//450:ids.append(appearance[ai]);ai+=1
        else:ids.append(other[oi]);oi+=1
    validate_permutation(batches,ids)
    return batches,ids

def validate_permutation(batches,ids):
    if len(batches)!=450 or any(len(b)!=6 for b in batches) or sorted(ids)!=list(range(450)):
        raise ValueError('Missing/duplicate/malformed batch')

def main():
    evidence();print('EVIDENCE_READY',OUT,flush=True)

if __name__=='__main__':main()
