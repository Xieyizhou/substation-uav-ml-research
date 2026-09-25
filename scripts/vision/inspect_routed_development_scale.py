"""Native-dimension scale matching; not viewpoint or visibility certification."""
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from scripts.vision.inspect_routed_transfer_coverage import OUT,bucket
from scripts.vision.routed_backbone_control import SOURCE,checked
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    pairs,_,deps=_load_inputs();by={r['image_sha256']:(r,t) for r,t in pairs};units=[];unique=Counter();details=[]
    for seed in (7,17,27):
        cp=SOURCE/'evaluation-v1/units'/f'material-routed-480-{seed}'/'completion.json';c=checked(cp);rp=Path(c['result']);result=checked(rp)
        counts=defaultdict(Counter)
        for row in result['rows']:
            if row['variant']!='material':continue
            source,truth=by[row['image_sha256']]
            if truth!=row['truth']:raise ValueError('Truth changed')
            with Image.open(source['image_path']) as im:w,h=im.size
            hits={x['truth_index'] for x in row['matches']}
            for i,t in enumerate(truth):
                b=bucket(t,w,h);key=t['class_name']+'|'+b
                counts[key]['truth']+=1;counts[key]['hit']+=int(i in hits)
                if seed==7:
                    unique[key]+=1
                    details.append(dict(pair_id=row['pair_id'],truth_index=i,truth=t,bucket=b,width=w,height=h,image_path=source['image_path'],image_sha256=row['image_sha256']))
        units.append(dict(seed=seed,counts={k:dict(v) for k,v in counts.items()}))
        deps.update({str(q.resolve()):file_sha256(q) for q in (cp,rp)})
    deps[str(Path(__file__).resolve())]=file_sha256(__file__)
    return write_record(OUT/'development-scale.json',dict(status='native_scale_verified_not_causal_diagnosis',unique_material_truths=dict(unique),units=units,details=details,training_admitted=False,promotable=False,inputs=deps))

if __name__=='__main__':
    r=run();print(r['unique_material_truths']);print(r['units'])
