"""Read existing full-label results; measure scale coverage without visual claims."""
from collections import Counter,defaultdict
from pathlib import Path
from scripts.vision.routed_backbone_control import SOURCE,checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=SOURCE.parent/'routed-transfer-coverage-v1'

def bucket(t,w,h):
    x1,y1,x2,y2=t['bbox_xyxy'];s=min(x2-x1,y2-y1)*640/max(w,h)
    return '<32' if s<32 else '32-64' if s<64 else '>=64'

def run():
    pp=SOURCE/'protocol.json';p=checked(pp);members={r['member_id']:r for r in p['pool_rows']}
    deps=[pp,Path(__file__)];training=Counter();exposure={};dev=defaultdict(Counter);events=[]
    for seed in (7,17,27):
        fp=SOURCE.parent/'routed-amplitude-fit-diagnosis-v1'/f'material-routed-480-{seed}.json';fit=checked(fp);deps.append(fp);counts=Counter()
        for r in fit['rows']:
            src=members[r['member_id']];truth=src['full_truth'];w,h=truth['image_width'],truth['image_height']
            for i,t in enumerate(r['truth']):
                key=t['class_name']+'|'+bucket(t,w,h)
                if seed==7:training[key]+=1
                counts[key]+=r['actual_exposures']
        exposure[str(seed)]=dict(counts)
        cp=SOURCE/'evaluation-v1/units'/f'material-routed-480-{seed}'/'completion.json';c=checked(cp);rp=Path(c['result']);r=checked(rp);deps += [cp,rp]
        # Resolve dimensions from the existing frozen paired development manifest.
        for row in r['rows']:
            if row['variant']!='material':continue
            for miss in row['misses']:
                i=miss['truth_index'];t=row['truth'][i]
                events.append(dict(seed=seed,pair_id=row['pair_id'],truth_index=i,truth=t,reason=miss['reason'],image_sha256=row['image_sha256']))
                dev[str(seed)][t['class_name']+'|'+miss['reason']]+=1
    OUT.mkdir(exist_ok=True)
    return write_record(OUT/'coverage.json',dict(status='numeric_coverage_only_visual_and_source_comparison_pending',training_material_instances_by_short_side_640=dict(training),actual_instance_exposure_by_seed=exposure,development_material_misses_by_class_reason={k:dict(v) for k,v in dev.items()},events=events,
        limits='Training scale from native dimensions; no inferred development dimensions. Same-source variants and seeds are not independent scenes. No visual content or viewpoint certification from boxes.',training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in deps}))

if __name__=='__main__':
    r=run();print(r['training_material_instances_by_short_side_640']);print(r['development_material_misses_by_class_reason'])
