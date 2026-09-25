"""Read-only source/condition coverage census; no inference or collection."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
from collections import Counter
from PIL import Image
from scripts.vision.check_reviewed_hold_fit import OUT as FIT,SOURCE,prior,base
from scripts.vision.structure_fit import OUT as TRACE

OUT=FIT.parent/'condition-view-coverage-v1'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def xml_value(e):
    if e is None:return None
    text=' '.join((e.text or '').split())
    try:text=[float(x) for x in text.split()] if text else None
    except ValueError:pass
    return [e.tag,dict(sorted(e.attrib.items())),text,[xml_value(c) for c in e]]

def world_fields(path):
    root=ET.parse(path).getroot();equipment=[]
    # Equipment identities come from the frozen scene names, not visual color.
    for m in root.findall('.//model'):
        name=m.get('name','')
        if not any(t in name for t in ('reactor','transformer','switchgear','capacitor')):continue
        visuals=[dict(name=v.get('name'),pose=xml_value(v.find('pose')),geometry=xml_value(v.find('geometry')),material=xml_value(v.find('material'))) for v in m.findall('.//visual')]
        equipment.append(dict(name=name,pose=xml_value(m.find('pose')),link_poses=[xml_value(l.find('pose')) for l in m.findall('link')],visuals=visuals))
    if not any(e['name']=='reactor_north' for e in equipment):raise ValueError('Expected reactor absent')
    geometry=[{**m,'visuals':[{k:v for k,v in x.items() if k!='material'} for x in m['visuals']]} for m in equipment]
    materials=[dict(model=m['name'],visuals=[dict(name=v['name'],material=v['material']) for v in m['visuals']]) for m in equipment]
    lighting=dict(ambient=xml_value(root.find('.//scene/ambient')),lights=[xml_value(l) for l in root.findall('.//light')])
    return dict(equipment_geometry_signature=digest(geometry),equipment_material_signature=digest(materials),lighting_signature=digest(lighting),
        equipment=equipment,lighting=lighting,background=xml_value(root.find('.//scene/background')),
        scope='Named equipment visual geometry, model/link/visual poses; not full collision or asset independence certification')

def bbox_features(box,w,h):
    x1,y1,x2,y2=box
    # YOLO decimal serialization may reconstruct zero as -9.6e-8 pixels.
    # Preserve the input; tolerate arithmetic noise only, never a one-pixel shift.
    if not (-1e-6<=x1<x2<=w+1e-6 and -1e-6<=y1<y2<=h+1e-6):raise ValueError('Invalid truth bounds')
    short=min(x2-x1,y2-y1)*640/max(w,h)
    return dict(short_side_640=short,size_bucket='lt32' if short<32 else '32to64' if short<64 else 'ge64',
        touches_image_edge=min(x1,y1,w-x2,h-y2)<=1,
        center_normalized=[(x1+x2)/(2*w),(y1+y2)/(2*h)],
        limitation='Box edge contact is not certified truncation or visibility; full_2d may include occluded extent')

def main():
    p=prior.read(FIT/'protocol.json');s=prior.read(FIT/'summary.json');c=prior.read(FIT/'crosscheck.json');trace=prior.read(TRACE/'member-source-trace.json')
    for r in (p,s,c,trace):prior.verify(r)
    paths=[FIT/'protocol.json',FIT/'summary.json',FIT/'crosscheck.json',FIT/'completion.json',TRACE/'member-source-trace.json',Path(__file__).resolve()]
    prior.verify(prior.read(FIT/'completion.json'));tr={r['member_id']:r for r in trace['rows']};idx={r['member_id']:r for r in p['rows']}
    common={m for m in idx if all(m in set(v['draws']) for v in p['models'].values())}
    reactor_mids={d['member_id'] for d in s['details'] if d['common_exposed'] and d['truth']['class_name']=='reactor'}
    selected=reactor_mids | (set(c['bridge_sources'])&common);worlds={};train=[];dev=[]
    def world(path):
        if not path.exists():return None
        sha=prior.file_sha256(path);paths.append(path)
        if sha not in worlds:worlds[sha]=dict(path=str(path),**world_fields(path))
        return sha
    for mid in sorted(selected):
        r=idx[mid];t=tr[mid];ip=Path(t['source_image'])
        if prior.file_sha256(ip)!=t['source_image_sha256']:raise ValueError('Original RGB changed')
        paths.append(ip);wp=ip.parents[2]/'plan/world.sdf';wh=world(wp)
        pose=t.get('actual_pose') or {};pose={k:pose.get(k) for k in ('position','orientation')}
        with Image.open(r['image_path']) as im:w,h=im.size
        labels=[dict(truth=v,**bbox_features(v['bbox_xyxy'],w,h)) for v in base.truth_for(r)]
        train.append(dict(member_id=mid,map_id=t['source_map'],source_view_id=t['source_view_id'],recording=t['recording'],world_sha256=wh,
            actual_carrier_pose=pose,registered_derivation=t['source_derivation_group'],registered_lineage=t['registered_lineage'],
            bridge_variant=c['bridge_sources'].get(mid,{}).get('variant','not_bridge'),labels=labels,
            prior_content_observation=t.get('reactor'),source_identity_gaps=t['gaps'],
            exposures={k:v['draws'].count(mid) for k,v in p['models'].items()}))
    sp=prior.read(SOURCE/'protocol.json');review_path=Path(sp['evaluation']['paired_review']);review=prior.read(review_path);prior.verify(review);paths.append(review_path)
    evalpath=SOURCE/'evaluation/brightness-450-7.json';evaluation=prior.read(evalpath);prior.verify(evaluation);paths.append(evalpath)
    ev={(r['view_id'],r['variant']):r for r in evaluation['rows']}
    for frame in review['frames']:
        ip=Path(frame['image_path']);wp=ip.parents[2]/'plan/world.sdf';wh=world(wp)
        if prior.file_sha256(ip)!=frame['image_sha256']:raise ValueError('Development RGB changed')
        paths.append(ip);receipt=ip.parents[1]/'collection-receipt.json';pose=None;gap=[]
        if receipt.exists():
            cr=prior.read(receipt);paths.append(receipt);vs=[v for v in cr['views'] if v['view_id']==frame['view_id']]
            if len(vs)!=1 or vs[0]['image_sha256']!=frame['image_sha256']:raise ValueError('Ambiguous dev receipt')
            pp=vs[0].get('actual_pose',{});pose={k:pp.get(k) for k in ('position','orientation')}
        else:gap.append('collection_receipt_not_at_image_capture_root')
        with Image.open(ip) as im:w,h=im.size
        labels=[dict(truth=v,**bbox_features(v['bbox_xyxy'],w,h)) for v in ev[frame['view_id'],frame['variant']]['truth']]
        dev.append(dict(view_id=frame['view_id'],pair_id=frame['pair_id'],variant=frame['variant'],world_sha256=wh,actual_carrier_pose=pose,labels=labels,gaps=gap))
    def reactor_summary(rows):
        labels=[l for r in rows for l in r['labels'] if l['truth']['class_name']=='reactor']
        return dict(instances=len(labels),size_buckets=dict(Counter(l['size_bucket'] for l in labels)),edge_contact=sum(l['touches_image_edge'] for l in labels))
    comparisons={}
    for variant in ('original','material','background','lighting'):
        ws={r['world_sha256'] for r in dev if r['variant']==variant}
        if len(ws)!=1 or None in ws:raise ValueError('Development world identity ambiguity')
        dw=worlds[next(iter(ws))];comparisons[variant]={}
        for field in ('equipment_geometry_signature','equipment_material_signature','lighting_signature'):
            matches=[r['member_id'] for r in train if r['world_sha256'] and worlds[r['world_sha256']][field]==dw[field]]
            comparisons[variant][field]=dict(matching_train_members=len(matches),members=matches)
    OUT.mkdir(exist_ok=True)
    prior.frozen(OUT/'matrix.json',dict(status='coverage_census_complete_not_causal_proof',training=train,development=dev,worlds=worlds,
        scope='Common-exposed reactor-bearing members plus common bridge positive members; not the entire training pool.',
        training_reactors=reactor_summary([r for r in train if r['member_id'] in reactor_mids]),
        development_reactors={v:reactor_summary([r for r in dev if r['variant']==v]) for v in ('original','material','background','lighting')},
        exact_component_matches=comparisons,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('MATRIX',len(train),len(dev),len(worlds),comparisons)

if __name__=='__main__':main()
