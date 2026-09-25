"""Source-aware visual error accounting; not an instance-mask certification."""
from collections import Counter,defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
from src.vision.canonical.plan import read_record
from scripts.vision.build_small_scale_fp_review import DEST,prior
from scripts.vision.record_small_scale_fp_review import run as review


def run():
    r=review(); ep=DEST/'evidence.json'; e=prior.read(ep); prior.verify(e)
    deps=[DEST/'review.json',ep,Path(__file__).resolve()]
    sources=[]; plans={}; per_cell=defaultdict(Counter); structure_groups=defaultdict(list)
    for d in r['decisions']:
        per_cell[d['cell']][d['visual_structure']]+=1
        structure_groups[d['image_id'],d['visual_structure']].append(d)
    for image in e['images']:
        src=image['source']; root=Path(src['image_path']).parents[2]/'plan'; pp=root/'plan.json'
        if str(pp) not in plans:
            p=read_record(pp); deps.append(pp)
            for name,sha in p['files'].items():
                f=root/name
                if prior.file_sha256(f)!=sha: raise ValueError('Source world/config drift')
                deps.append(f)
            world=ET.parse(root/'world.sdf').getroot().find('world')
            models={m.get('name'):m for m in world.findall('model')}
            assets=[]
            for name in ('cabinet_center','control_building'):
                m=models[name]; obj=next(o for o in p['objects'] if o['name']==name)
                assets.append(dict(name=name,source_category=obj['category'],pose=m.findtext('pose'),
                    materials={v.get('name'):v.findtext('material/diffuse') for v in m.findall('.//visual')}))
            target_names={o['name'] for o in p['objects'] if o['category'] in ('transformer','switchgear','capacitor_bank','reactor')}
            if target_names & set(models): raise ValueError('Target models remain in negative world')
            plans[str(pp)]=dict(plan_identity=p['identity'],assets=assets,target_models_absent=True,plan=p)
        p=plans[str(pp)]['plan']
        views={}
        for v in p.get('calibration_views',[])+p.get('pilot_views',[]):
            if v['view_id'] in views and views[v['view_id']]!=v: raise ValueError('Conflicting view identity')
            views[v['view_id']]=v
        v=views[src['view_id']]
        if v['lighting_id']!=src['variant']: raise ValueError('Source lighting mismatch')
        sources.append(dict(image_id=image['image_id'],image_sha256=src['image_sha256'],plan_path=str(pp),
            pose_id=v['pose_id'],derivation_group=v['derivation_group'],source_layout_id=p['source_layout_id'],
            planned_subject=v['object_id'],note='Planned subject is not assigned to individual prediction boxes.'))
    dest=DEST/'analysis.json'
    if dest.exists():
        old=prior.read(dest); prior.verify(old); return old
    return prior.frozen(dest,dict(status='source_and_visual_accounting_complete',sources=sources,
        source_worlds={k:{f:v for f,v in p.items() if f!='plan'} for k,p in plans.items()},
        counts_by_cell={k:dict(v) for k,v in per_cell.items()},
        same_image_structure_groups=[dict(image_id=k[0],visual_structure=k[1],events=[d['event_id'] for d in ds],
            seeds=sorted({d['seed'] for d in ds}),cells=sorted({d['cell'] for d in ds}),
            interpretation='Same image and reviewed visual content category; not a cross-view asset-instance match.') for k,ds in structure_groups.items()],
        unique_pose_ids=len({s['pose_id'] for s in sources}),unique_images=len(sources),
        limitations=['Appearance categories are explicit visual observations; no per-pixel instance mask has been inferred.',
            'Mixed-structure groups may contain different overlapping components; not counted as one physical object.',
            'Known source assets and layouts repeat across poses and lighting; seeds do not add independent samples.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__':
    r=run(); print(r['status'],r['unique_images'],r['unique_pose_ids'],r['counts_by_cell'])
