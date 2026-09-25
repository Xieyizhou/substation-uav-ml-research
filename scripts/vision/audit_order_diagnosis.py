"""Verify runtime transforms and compare every arm directly with the single reference."""
import sys
from pathlib import Path
from collections import Counter
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_order_diagnosis import OUT,REFERENCE,KEYS,batches,validate_transforms,read,save,file_sha256,verify_tree,object_sha256

def main():
    verify_tree(OUT/'completion.json');p=read(OUT/'protocol.json');c=read(OUT/'completion.json')
    refcell=read(REFERENCE/'F-100-7/completion.json');original=read(refcell['exposure_path'])['draws']
    refargs=yaml.safe_load((Path(refcell['exposure_path']).parent/'args.yaml').read_text())
    actuals={};audits={}
    for k in KEYS:
        cell=read(OUT/k/'completion.json');actual=read(cell['exposure_path']);actuals[k]=actual['draws']
        if actuals[k]!=p['schedules'][k] or actual['summary']!=p['exposures'][k] or cell['optimizer_steps']!=100:raise ValueError('Runtime exposure mismatch')
        args=yaml.safe_load((Path(cell['exposure_path']).parent/'args.yaml').read_text())
        if any(args[n]!=v for n,v in refargs.items() if n not in ('name','project','save_dir')):raise ValueError('Training controls differ')
        ep=OUT/f'evaluation-{k}.json';r=read(ep);pairs=[]
        for pair_id in sorted({x['pair_id'] for x in r['rows']}):
            group={x['variant']:x for x in r['rows'] if x['pair_id']==pair_id}
            if set(group)!=set(('original','material','background','lighting')):raise ValueError('Incomplete pair')
            for v in ('material','background','lighting'):
                a,b=group['original'],group[v]
                pairs.append(dict(pair_id=pair_id,variant=v,planned_hit_delta=int(b['planned_instance_hit'])-int(a['planned_instance_hit']),matched_instance_delta=len(b['matches'])-len(a['matches'])))
        audits[k]=dict(sequence_sha256=object_sha256(actuals[k]),batch_sha256=object_sha256(batches(actuals[k])),paired_variant_deltas=pairs,
            same_batch_contents_in_same_order=all(Counter(a)==Counter(b) for a,b in zip(batches(original),batches(actuals[k]))),
            same_ordered_batch_collection=Counter(map(tuple,batches(original)))==Counter(map(tuple,batches(actuals[k]))),
            max_parameter_abs_difference=max(d['max_abs'] for d in r['parameter_differences'].values()),
            negative_summary=r['negative_summary'],summary=r['summary'])
    validate_transforms(original,actuals)
    result=save(OUT/'audit.json',dict(status='passed',actual_transform_constraints_verified=True,training_settings_equal=True,arms=audits,
        inputs={str(q):file_sha256(q) for q in (Path(__file__),OUT/'completion.json',REFERENCE/'F-100-7/completion.json')}))
    for k,a in audits.items():
        print(k,a['negative_summary'],'parameter_max_delta',a['max_parameter_abs_difference'])
        print({v:{m:s[m] for m in ('planned_instance_hit_rate','instance_recall','matched_precision','unmatched_predictions')} for v,s in a['summary'].items()})
        print('material_checks',[x for x in c['comparisons'][k]['checks'] if x['material']])
    print(c['tests_output'])

if __name__=='__main__':main()
