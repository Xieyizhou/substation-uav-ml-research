"""Independent post-training audit; explicit visual decisions are separate."""
import csv
import math
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image
from scripts.vision.exposure_order_retention import OUT as SOURCE,PRIOR,KEYS,ROOT,read,save,file_sha256,verify_tree,identity,panel,frozen
from scripts.vision.run_exposure_diagnosis import checked_cell
from scripts.vision.run_matched_appearance_training import validate_evaluation
from scripts.vision.prepare_retention450_training import validate_args
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks

OUT=SOURCE/'post-training-review-v1'

def build():
    dest=OUT/'manifest.json'
    if dest.exists():verify_tree(dest);return read(dest)
    OUT.mkdir(exist_ok=True);p=read(SOURCE/'protocol.json');records={};inputs={}
    for path in (SOURCE/'ready.json',SOURCE/'evaluation-path-repair-v1.json'):
        verify_tree(path);inputs[str(path)]=file_sha256(path)
    np=Path(p['evaluation']['negative_review']);pp=Path(p['evaluation']['paired_review'])
    nmap={(x['view_id'],x['variant']):x for x in read(np)['frames']};pmap={(x['view_id'],x['variant']):x for x in read(pp)['frames']}
    for path in (np,pp):inputs[str(path)]=file_sha256(path)
    neg=[];losses=[]
    for key in KEYS:
        cp=SOURCE/'training'/key/'completion.json';cell=checked_cell(cp,p)
        if cell['optimizer_steps']!=450:raise ValueError('Wrong endpoint')
        folder=Path(cell['exposure_path']).parent;args=yaml.safe_load((folder/'args.yaml').read_text());validate_args(args)
        if args['seed']!=int(key.split('-')[-1]) or args['model']!=p['initialization']['path']:raise ValueError('Seed / initialization mismatch')
        curves=list(csv.DictReader((folder/'results.csv').open()))
        if len(curves)!=45:raise ValueError('Incomplete curves')
        for row in curves:
            if not any(k.strip().startswith('lr/') for k in row):raise ValueError('Missing LR curve')
            for k,v in row.items():
                if k.strip().startswith('train/') and not math.isfinite(float(v)):raise ValueError('Invalid loss')
                if k.strip().startswith('lr/') and abs(float(v)-.001)>1e-12:raise ValueError('LR changed')
        ep=SOURCE/'training'/f'evaluation-{key}.json';r=read(ep);validate_evaluation(r,key);records[key]=r
        for path,digest in r['inputs'].items():
            if file_sha256(path)!=digest:raise ValueError('Stale evaluation')
        for path in (cp,ep,folder/'args.yaml',folder/'results.csv'):inputs[str(path)]=file_sha256(path)
        for row in r['negative_rows']:
            src=nmap[row['view_id'],row['variant']];ip=Path(src['image_path'])
            if file_sha256(ip)!=row['image_sha256']:raise ValueError('Stale negative pixels')
            for index,pred in enumerate(row['predictions']):
                eid=f'F{len(neg)+1:02}';path=OUT/f'{eid}.png'
                panel(ip,pred['bbox_xyxy'],path,f'{eid} {key} {row["variant"]} {pred["class_name"]} {pred["confidence"]:.4f}')
                neg.append(dict(event_id=eid,cell=key,view_id=row['view_id'],variant=row['variant'],prediction_index=index,prediction=pred,
                    image_path=str(ip),image_sha256=row['image_sha256'],evidence_path=str(path),evidence_sha256=file_sha256(path)))
                inputs[str(ip)]=file_sha256(ip);inputs[str(path)]=file_sha256(path)
    # Inspect every newly missed original/lighting reactor versus fixed ability reference,
    # not merely planned targets or losses versus the staged comparator.
    refs=[]
    for seed in (7,17,27):
        rp=PRIOR/f'evaluation-retained_reference-450-{seed}.json';ref=read(rp);refs.append(ref);inputs[str(rp)]=file_sha256(rp)
        old={(x['pair_id'],x['variant']):x for x in ref['rows']}
        for row in records[f'interleaved-450-{seed}']['rows']:
            if row['variant'] not in ('original','lighting'):continue
            rr=old[row['pair_id'],row['variant']];a={identity(t):i for i,t in enumerate(rr['truth'])};b={identity(t):i for i,t in enumerate(row['truth'])}
            if set(a)!=set(b) or len(a)!=len(rr['truth']) or len(b)!=len(row['truth']):raise ValueError('Truth identity conflict')
            for ident,index in b.items():
                if ident[1]!='reactor':continue
                if not any(m['truth_index']==a[ident] for m in rr['matches']) or any(m['truth_index']==index for m in row['matches']):continue
                eid=f'R{len(losses)+1:02}';ip=Path(pmap[row['view_id'],row['variant']]['image_path']);path=OUT/f'{eid}.png'
                panel(ip,row['truth'][index]['bbox_xyxy'],path,f'{eid} interleaved seed {seed} {row["variant"]} reactor {ident[0]}',row['predictions'])
                losses.append(dict(event_id=eid,seed=seed,variant=row['variant'],pair_id=row['pair_id'],instance_label=ident[0],
                    truth=row['truth'][index],misses=[m for m in row['misses'] if m['truth_index']==index],low_predictions=row['low_predictions'],
                    image_path=str(ip),image_sha256=file_sha256(ip),evidence_path=str(path),evidence_sha256=file_sha256(path)))
                inputs[str(ip)]=file_sha256(ip);inputs[str(path)]=file_sha256(path)
    if len(neg)!=42:raise ValueError('Unexpected FP scope')
    groups={arm:aggregate([records[f'{arm}-450-{s}'] for s in (7,17,27)]) for arm in ('staged','interleaved')}
    hp=Path(p['evaluation']['historical_reference']);hist=read(hp);inputs[str(hp)]=file_sha256(hp)
    checks={arm:policy_checks(g,aggregate(refs),hist['historical_A'],p) for arm,g in groups.items()}
    for g in checks.values():
        for c in g['checks']:
            if c.get('reference')=='same_budget_R':c['reference']='retained_reference-450'
    for offset in range(0,len(neg),3):
        page=Image.new('RGB',(1200,1620),'white')
        for i,e in enumerate(neg[offset:offset+3]):page.paste(Image.open(e['evidence_path']),(0,i*540))
        path=OUT/f'page-{offset//3+1:02}.png';page.save(path);inputs[str(path)]=file_sha256(path)
    inputs[str(Path(__file__))]=file_sha256(Path(__file__))
    result=frozen(dest,dict(status='numerically_rejected_visual_review_pending',negative=neg,reactor_losses=losses,
        aggregate=groups,policy=checks,units_verified=6,inputs=inputs))
    print('MANIFEST_READY',len(neg),'FPs',len({e['image_sha256'] for e in neg}),'images',len(losses),'reactor losses',OUT,flush=True)
    return result

if __name__=='__main__':build()
