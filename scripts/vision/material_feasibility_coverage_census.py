"""Actual-exposure census with per-label saved material, never planned-class counts."""
from collections import Counter
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from scripts.vision.material_control_feasibility import OUT,RUN,CANDIDATE,prior
from scripts.vision.test_body_material_applicability import source_index,model_mapping
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence
from scripts.vision.structure_fit import truth_for,pixels

def main():
    p=prior.read(RUN/'protocol.json');prior.verify(p)
    sources,paths=source_index();paths += [RUN/'protocol.json',OUT/'risk-containment.json',Path(__file__).resolve()]
    lightpath=RUN.parent/'light-review/evidence.json';designpath=RUN.parent/'design.json'
    light,design=prior.read(lightpath),prior.read(designpath)
    for x in (light,design):prior.verify(x)
    paths += [lightpath,designpath]
    rows=[];cache={}
    for m in p['pool_rows']:
        for field in ('image','label'):
            if prior.file_sha256(m[field+'_path'])!=m[field+'_sha256']:raise ValueError('Changed pool input')
            paths.append(Path(m[field+'_path']))
        if m['variant']=='physical-lighting':
            actual=[x for x in light['events'] if x['pair_id']==m['pair_id']]
            if not actual or len({x['image'] for x in actual})!=1:raise ValueError('Ambiguous light derivative origin')
            equal_rgb(m['image_path'],actual[0]['image']);paths.append(Path(actual[0]['image']))
        else:equal_rgb(m['image_path'],m['source_image_path'])
        paths.append(Path(m['source_image_path']))
        if prior.file_sha256(m['source_label_path'])!=m['label_sha256']:raise ValueError('Changed full export label')
        paths.append(Path(m['source_label_path']));truth=truth_for(m)
        if Counter(t['class_name'] for t in truth)!=Counter(m['class_instances']):raise ValueError('Full class counts differ')
        r=dict(member_id=m['member_id'],subset=m['subset'],lineage_id=m['lineage_id'],
            class_instances=m['class_instances'],wrapper_variant=m['variant'],native_pixels_verified=True,
            actual_exposures={k:Counter(v)[m['member_id']] for k,v in p['schedules'].items()},instances=[],gaps=[])
        source=sources.get(m.get('source_member_id',m['member_id']))
        if source is None:
            r['gaps'].append('original_capture_not_in_positive_source_index' if truth else 'negative_source_world_not_censused_no_target_labels')
        elif m['variant']=='physical-lighting':
            f=next(x for x in design['light_sources'] if x['member_id']==m['source_member_id'])
            wp=Path(f['lighting_world']);paths.append(wp)
            _,models=model_mapping(ET.parse(wp))
            original=next(x for x in rows if x['member_id']==m['source_member_id'])
            r.update(source_variant='physical-lighting',world_path=str(wp),world_sha256=prior.file_sha256(wp),actual_pose=f['actual_pose'],
                source_status='registered_light_derivative_RGB_verified_against_own_capture',gaps=list(original['gaps']))
            for t in original['instances']:
                materials={v.get('name'):{k:v.findtext('material/'+k) for k in ('ambient','diffuse')} for v in models[t['object_id']].findall('.//visual')}
                r['instances'].append(dict(t,materials=materials))
        else:
            ip=Path(source.get('source_image_path',source.get('image_path')))
            try:
                equal_rgb(ip,m['image_path']);paths.append(ip)
                cp=ip.parents[1]/'collection-receipt.json';pp=ip.parents[2]/'plan/plan.json';wp=pp.parent/'world.sdf'
                if not all(q.exists() for q in (cp,pp,wp)):raise ValueError('Original capture/plan/world unavailable')
                if cp not in cache:
                    rec,plan=prior.read(cp),prior.read(pp);digest=prior.file_sha256(wp)
                    if plan.get('files',{}).get('world.sdf')!=digest or rec.get('world_sha256') not in (None,digest):raise ValueError('World hash conflict')
                    mapping,models=model_mapping(ET.parse(wp));cache[cp]=(rec,mapping,models)
                rec,mapping,models=cache[cp];paths += [cp,pp,wp]
                views=[v for v in rec['views'] if v['view_id']==ip.parent.name]
                if len(views)!=1 or views[0]['image_sha256']!=prior.file_sha256(ip):raise ValueError('Capture RGB identity conflict')
                raw=label_correspondence(truth,views[0]['truth']['objects'])
                r.update(source_variant=source.get('variant','unknown'),world_path=str(wp),world_sha256=prior.file_sha256(wp),
                    source_capture=str(cp),source_view_id=ip.parent.name,actual_pose=views[0].get('actual_pose'),source_status='posthoc_saved_world_trace')
                for t in raw:
                    label=str(int(re.search(r'instance-(\d+)-',t['annotation_id'])[1]));name=mapping[label]
                    visuals=models[name].findall('.//visual')
                    materials={v.get('name'):{k:v.findtext('material/'+k) for k in ('ambient','diffuse')} for v in visuals}
                    r['instances'].append(dict(object_id=name,class_name=t['class_name'],bbox_xyxy=t['bbox_xyxy'],materials=materials))
                if not rec.get('world_sha256'):r['gaps'].append('historical_actual_world_hash_not_recorded')
                if not rec.get('collection_checks',{}).get('instance_mapping'):r['gaps'].append('historical_actual_mapping_not_recorded')
            except (ValueError,KeyError,AttributeError,IndexError) as ex:
                r['gaps'].append('source_trace_blocked: '+str(ex))
        rows.append(r)
    cp=CANDIDATE/'protocol.json';sp=CANDIDATE/'semantic-review.json';ep=CANDIDATE/'review-evidence.json'
    candidate,semantic,evidence=map(prior.read,(cp,sp,ep))
    for x in (candidate,semantic,evidence):prior.verify(x)
    paths += [cp,sp,ep]
    candidates=[]
    dev=prior.read(OUT/'initial-gate.json')['corrected_target_records']
    dev_files={x['image_sha256'] for x in dev};dev_pixels={pixels(x['image_path']) for x in dev}
    for frame in semantic['frames']:
        src=next(f for f in candidate['frames'] if frame['source_review_id'] in f['review_ids'])
        events=[x for x in evidence['events'] if (x['source_review_id'],x['variant'])==(frame['source_review_id'],frame['variant'])]
        if not events:raise ValueError('Missing candidate full labels')
        for ev in events:
            for field in ('image','crop','full_context'):
                q=Path(ev[field+'_path'])
                if prior.file_sha256(q)!=ev[field+'_sha256']:raise ValueError('Stale candidate evidence')
                paths.append(q)
        im=events[0]['image_path']
        if prior.file_sha256(im) in dev_files or pixels(im) in dev_pixels:raise ValueError('Candidate/development duplicate')
        classes=Counter(x['source_truth']['class_name'] for x in events)
        source_row=next((x for x in rows if x['member_id']==src['member_id']),None)
        if source_row and classes!=Counter(source_row['class_instances']):raise ValueError('Candidate complete class count differs')
        candidates.append(dict(**frame,member_id=src['member_id'],lineage_id=src['lineage_id'],full_label_classes=dict(classes),
            altered_target_objects=[x['object_id'] for x in src['events']],altered_target_class=src['class_name'],
            actual_source_exposures=source_row['actual_exposures'] if source_row else {},
            file_and_native_pixel_exclusion_passed=True,source_independent=False,
            independence_note='Same complex layout/assets as development; new pixel hashes do not prove source isolation.',
            training_ready=False,disposition='held_do_not_restore' if frame['status']!='reviewed_candidate_only' else 'quality_reviewed_bounded_candidate_not_training_approved'))
    totals={}
    for k in p['schedules']:
        counts=Counter()
        for r in rows:
            for cat,n in r['class_instances'].items():counts[cat]+=n*r['actual_exposures'][k]
        totals[k]=dict(counts)
    prior.frozen(OUT/'coverage-census.json',dict(status='census_complete_with_explicit_source_gaps',members=rows,
        candidates=candidates,class_instance_exposure=totals,
        unresolved_positive_sources=[dict(member_id=r['member_id'],gaps=r['gaps']) for r in rows if r['class_instances'] and r['gaps']],
        scope='Current 240 pool rows and eight frozen closed-body candidates; older candidate collections are not automatically approved.',
        coverage_certified_sufficient=False,training_started=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('CENSUS',len(rows),'resolved instances',sum(len(r['instances']) for r in rows),'source failures',sum(any(g.startswith('source_trace_blocked') for g in r['gaps']) for r in rows))

if __name__=='__main__':main()
