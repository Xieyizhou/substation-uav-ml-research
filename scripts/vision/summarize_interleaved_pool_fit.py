"""Summarize complete fixed diagnostic units without converting to admission."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_interleaved_pool_fit import OUT,KEYS,prior,freeze,runtime

def main():
    p=freeze();stats={};deps=[OUT/'protocol.json',Path(__file__).resolve()]
    for key in KEYS:
        path=OUT/(key+'.json');r=prior.read(path);runtime.validate(r,key,p);deps.append(path)
        groups={}
        for m,row in zip(p['members'],r['rows']):
            count=p['actual_exposures'][key].get(m['member_id'],0)
            group=('exposed' if count else 'unexposed')+':'+m.get('variant','unknown')
            g=groups.setdefault(group,dict(images=0,image_exposures=0,truth=0,matched=0,unmatched_predictions=0,classes={}))
            g['images']+=1;g['image_exposures']+=count;g['truth']+=len(row['truth']);g['matched']+=len(row['matches']);g['unmatched_predictions']+=row['unmatched_prediction_count']
            hits={v['truth_index'] for v in row['matches']}
            for j,t in enumerate(row['truth']):
                c=g['classes'].setdefault(t['class_name'],dict(truth=0,matched=0,instance_exposures=0,miss_reasons={}))
                c['truth']+=1;c['matched']+=int(j in hits);c['instance_exposures']+=count
                if j not in hits:
                    reason=next(v['reason'] for v in row['misses'] if v['truth_index']==j)
                    c['miss_reasons'][reason]=c['miss_reasons'].get(reason,0)+1
        stats[key]=groups
    result=prior.frozen(OUT/'summary.json',dict(status='training_fit_diagnostic_complete_not_generalization',statistics=stats,inputs={str(d):prior.file_sha256(d) for d in deps}))
    for key,groups in stats.items():
        for name,g in groups.items():print(key,name,g['images'],g['matched'],g['truth'],g['unmatched_predictions'])
    return result
if __name__=='__main__':main()
