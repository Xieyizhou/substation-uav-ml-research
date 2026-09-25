"""Trace existing training-candidate sources; never restore holds or approve images."""
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from scripts.vision.test_body_material_applicability import source_index
from scripts.vision.train_frozen_multiscale import contract,prior
from scripts.vision.material_control_feasibility import OUT as HISTORY
from src.vision.canonical.gates import validate_preflight,validate_point
from src.vision.canonical.plan import rotate

OUT=HISTORY.parent/'material-view-candidates-v1'

def main():
    sources,paths=source_index();_,p,_=contract('fixed-7')
    risk=prior.read(HISTORY/'risk-containment.json');prior.verify(risk);paths.append(HISTORY/'risk-containment.json')
    seen={mid for seq in p['schedules'].values() for mid in seq}
    aliases={m.get('source_member_id',m['member_id']) for m in p['pool_rows'] if m['member_id'] in seen}
    seen|=aliases
    held=set(p['held_members'])|set(risk['denied_members'])
    seen_groups={sources[mid].get('derivation_group') or sources[mid].get('view_id') for mid in seen if mid in sources}
    held_groups={sources[mid].get('derivation_group') or sources[mid].get('view_id') for mid in held if mid in sources}
    held_groups.update(risk['denied_lineages'])
    records=[];cache={}
    for mid,row in sorted(sources.items()):
        if not mid.startswith('candidate:'):continue
        group=row.get('derivation_group');r=dict(source_member_id=mid,lineage_id=group,source_image=row['image_path'],status='blocked',gaps=[])
        records.append(r)
        if mid in held or group in held_groups:r['gaps'].append('existing_held_source_or_lineage');continue
        if mid in seen or group in seen_groups:r['gaps'].append('source_or_registered_pose_already_exposed');continue
        if row.get('review_decision')!='accepted':r['gaps'].append('historical_review_not_accepted');continue
        ip=Path(row['image_path']);cp=ip.parents[1]/'collection-receipt.json';pp=ip.parents[2]/'plan/plan.json';wp=pp.parent/'world.sdf'
        if not all(x.exists() for x in (ip,cp,pp,wp)):r['gaps'].append('missing_saved_source_bundle');continue
        try:
            if prior.file_sha256(ip)!=row['image_sha256']:raise ValueError('source_image_hash_changed')
            if cp not in cache:cache[cp]=(prior.read(cp),prior.read(pp))
            receipt,plan=cache[cp]
            views=[x for k in ('calibration_views','pilot_views') for x in plan.get(k,[]) if x['view_id']==ip.parent.name]
            if len(views)!=1:raise ValueError('nonunique_planned_view')
            view=views[0];actual=[x for x in receipt['views'] if x['view_id']==view['view_id']]
            if len(actual)!=1 or actual[0]['status']!='captured':raise ValueError('missing_successful_capture')
            actual=actual[0]
            if view['category'] not in ('transformer','switchgear','capacitor_bank','reactor'):raise ValueError('not_positive_target_view')
            checks,config,mapping=validate_preflight(plan,pp.parent,[view])
            if actual['image_sha256']!=row['image_sha256']:raise ValueError('receipt_image_mismatch')
            if receipt.get('world_sha256')!=checks['world_sha256']:raise ValueError('actual_world_hash_missing_or_changed')
            if receipt.get('collection_checks',{}).get('instance_mapping')!=checks['instance_mapping']:raise ValueError('actual_instance_mapping_missing_or_changed')
            pose=actual['actual_pose'];validate_point(pose['position'],config,role='saved_actual_carrier')
            cam=ET.parse(wp).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
            offset=list(map(float,cam.findtext('pose').split()))[:3];delta=rotate(pose['orientation'],offset)
            optical=[a+b for a,b in zip(pose['position'],delta)];validate_point(optical,config,role='saved_actual_optical')
            r.update(status='source_trace_passed_review_required',category=view['category'],object_id=view['object_id'],actual_pose=pose,
                source_plan=str(pp),source_world=str(wp),source_capture=str(cp),source_view_id=view['view_id'],
                world_name=plan['world_name'],material_id=row.get('material_id'),lighting_id=row.get('lighting_id'),
                image_sha256=row['image_sha256'],source_record=actual,instance_mapping=checks['instance_mapping'])
            paths += [ip,cp,pp,wp,pp.parent/'obstacles.json']
        except (ValueError,KeyError,TypeError,AttributeError) as ex:r['gaps'].append(str(ex))
    OUT.mkdir(exist_ok=True)
    # New view status remains provisional until physical-pose/lineage checks and fresh full-frame review.
    prior.frozen(OUT/'source-inventory.json',dict(status='source_inventory_complete_no_automatic_review',records=records,
        rejection_counts=dict(Counter(g for r in records for g in r['gaps'])),
        passed_frames=sum(r['status']=='source_trace_passed_review_required' for r in records),
        training_ready=False,training_started=False,collection_started=False,
        inputs={str(x):prior.file_sha256(x) for x in paths+[Path(__file__).resolve()]}))
    print('COUNTS',Counter(r['status'] for r in records),Counter(g for r in records for g in r['gaps']))
    for r in records:
        if r['status']=='source_trace_passed_review_required':print(r['category'],r['lineage_id'],r['material_id'],r['lighting_id'])

if __name__=='__main__':main()
