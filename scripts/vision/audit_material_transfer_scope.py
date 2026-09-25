"""Actual saved-world scope and changed-instance exposure audit. No rendering/training."""
from collections import Counter, defaultdict
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from scripts.vision import double_dose_pool_fit as fit
from scripts.vision import compensated_pool_fit as low
from scripts.vision.merge_compensated_material_candidates import OUT as CAND
from scripts.vision.build_material_view_world_drafts import check_only_materials, material_nodes, PALETTES
from scripts.vision.prepare_paired_visual_factors import BASE, assert_allowed_world_diff

OUT=CAND/'condition-transfer-route-v1'
prior=fit.prior


def resolve_truth(truth, mapping):
    objects=[v['object_id'] for v in mapping.values()]
    if len(objects)!=len(set(objects)): raise ValueError('Instance mapping collision')
    result=[]
    for t in truth['objects']:
        match=re.search(r'-instance-(\d+)-',t['annotation_id'])
        if not match: raise ValueError('Unresolved truth instance')
        entry=mapping.get(str(int(match.group(1))))
        if entry is None or entry['category']!=t['class_name']: raise ValueError('Missing/mismatched instance')
        result.append(dict(object_id=entry['object_id'], category=entry['category'], bbox_xyxy=t['bbox_xyxy']))
    if len({x['object_id'] for x in result})!=len(result): raise ValueError('Duplicate truth instance')
    return result


def run():
    cp=CAND/'reviewed-completion.json'; candidates=prior.read(cp); prior.verify(candidates)
    hp=prior.read(fit.OUT/'protocol.json'); lp=prior.read(low.OUT/'protocol.json')
    fit.check(hp); low.check(lp)
    metadata={m['member_id']:m for m in hp['members']}
    paths=[cp,fit.OUT/'protocol.json',low.OUT/'protocol.json',Path(__file__).resolve()]
    rows=[]; index={}
    for m in candidates['members']:
        if m['variant']=='original':continue
        source=Path(m['source_review']).parent/m['source_pose_id']/'protocol.json'
        sp=prior.read(source);prior.verify(sp)
        cap=Path(m['variant_review']).parent/'protocol.json'; capture=prior.read(cap);prior.verify(capture)
        before=Path(sp['frame']['source_world']); after=Path(m['world_path'])
        for path in (before,after):
            if capture['inputs'].get(str(path))!=prior.file_sha256(path): raise ValueError('World not bound to capture')
        a,b=ET.parse(before).getroot(),ET.parse(after).getroot()
        target=m['planned_object_id']
        changed=check_only_materials(a,b,{target},('body','reactor'))
        if len(changed)!=2:raise ValueError('Unexpected change count')
        nodes=material_nodes(b,{target},('body','reactor'))
        if any(n.text.split()!=PALETTES[m['variant']].split() for n in nodes.values()):raise ValueError('Palette mismatch')
        truth=resolve_truth(m['full_truth'],m['instance_mapping'])
        if sum(t['object_id']==target for t in truth)!=1:raise ValueError('Planned material target absent')
        normalized=metadata[m['member_id']]['truth']
        if len(normalized)!=len(truth):raise ValueError('Full label count mismatch')
        for t,n in zip(truth,normalized):
            if t['category']!=n['class_name'] or max(abs(x-y) for x,y in zip(t['bbox_xyxy'],n['bbox_xyxy']))>1e-5:
                raise ValueError('Normalized label correspondence mismatch')
        row=dict(member_id=m['member_id'],source_pose_id=m['source_pose_id'],pair_id=m['pair_id'],variant=m['variant'],
                 changed_object_id=target,changed_fields=changed,truth=truth,world_path=str(after),original_world=str(before))
        rows.append(row);index[m['member_id']]=row
        paths += [source,cap,before,after]
    if len(rows)!=24:raise ValueError('Missing material member')
    original=BASE/'original/plan/world.sdf'; material=BASE/'material/plan/world.sdf'
    for variant,path in [('original',original),('material',material)]:
        pp=path.parent/'plan.json';p=prior.read(pp);prior.verify(p)
        if p['files']['world.sdf']!=prior.file_sha256(path):raise ValueError('Development world hash mismatch')
        paths += [pp,path]
    assert_allowed_world_diff(original,material,'material')
    a,b=ET.parse(original).getroot(),ET.parse(material).getroot()
    names={v['object_id'] for v in candidates['members'][0]['instance_mapping'].values()}
    dev_changes=check_only_materials(a,b,names)
    bn=material_nodes(b,names)
    changes=[dict(object_id=k[0],link=k[1],visual=k[2],field=k[3],value=bn[tuple(k)].text) for k in dev_changes]
    results={}
    for key in fit.KEYS:
        g=defaultdict(Counter)
        for dose,folder,p in [('low',low.OUT,lp),('high',fit.OUT,hp)]:
            path=folder/'inference'/f'{key}.json';r=prior.read(path);fit.validate_unit(r,key,p);paths.append(path)
            for item in r['rows']:
                if item['member_id'] not in index or item['actual_exposures']==0:continue
                source=index[item['member_id']];hits={x['truth_index'] for x in item['matches']}
                for i,t in enumerate(source['truth']):
                    altered=t['object_id']==source['changed_object_id']
                    tag='changed_body' if altered else 'unchanged_incidental'
                    c=g[dose,tag,t['category']]
                    c['unique_instance_rows']+=1;c['hits']+=i in hits;c['actual_instance_exposures']+=item['actual_exposures']
        results[key]=[dict(dose=d,scope=s,category=c,**v) for (d,s,c),v in sorted(g.items())]
    body=dict(status='saved_world_scope_and_changed_instance_fit_verified',material_members=rows,
              development_changes=changes,training_changed_objects=sorted({r['changed_object_id'] for r in rows}),
              development_changed_objects=sorted({x['object_id'] for x in changes}),fit_by_actual_intervention=results,
              training_started=False,rendering_started=False,training_ready=False,
              limits=['World fields, not measured causal feature dependence',
                      'Changed-body labels differ from all labels in a material image',
                      'Shared layout/assets and derived variants are not independent sources',
                      'No historical supervision issue is cleared by this audit'],
              inputs={str(p):prior.file_sha256(p) for p in paths})
    OUT.mkdir(exist_ok=True);dest=OUT/'scope-audit.json'
    if dest.exists():
        old=prior.read(dest);prior.verify(old)
        if any(old[k]!=v for k,v in body.items()):raise ValueError('Audit drift')
        return old
    return prior.frozen(dest,body)


if __name__=='__main__':
    r=run();print(r['status']);print('DEV',len(r['development_changes']),'fields',len(r['development_changed_objects']),'objects')
    for k,v in r['fit_by_actual_intervention'].items():print(k,v)
