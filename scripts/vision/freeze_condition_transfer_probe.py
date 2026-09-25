"""Freeze source-pose factorial material diagnostic drafts. Never capture or train."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision import audit_material_transfer_scope as audit
from scripts.vision.build_material_view_world_drafts import material_nodes, check_only_materials, signature

OUT=audit.OUT/'factorial-probe-v1'
CONDITIONS={
    'gray_target_body': ('target', ('body','reactor')),
    'gray_target_full': ('target', ('body','reactor','front_panel')),
    'gray_all_body': ('all', ('body','reactor')),
    'gray_all_full': ('all', ('body','reactor','front_panel')),
}


def derive(base, target, names, condition):
    scope,visuals=CONDITIONS[condition]
    allowed={target} if scope=='target' else set(names)
    if target not in names: raise ValueError('Target outside instance mapping')
    result=copy.deepcopy(base);nodes=material_nodes(result,allowed,visuals)
    if not nodes:raise ValueError('No eligible materials')
    for n in nodes.values(): n.text='0.35 0.35 0.35 1'
    changes=check_only_materials(base,result,allowed,visuals)
    return result,changes


def freeze():
    a=audit.run();prior=audit.prior
    paths=[audit.OUT/'scope-audit.json',Path(__file__).resolve()]
    candidates=prior.read(audit.CAND/'reviewed-completion.json')
    chosen=sorted([m for m in candidates['members'] if m['variant']=='warm'],key=lambda m:m['member_id'])
    OUT.mkdir(exist_ok=True);worlds=OUT/'worlds';worlds.mkdir(exist_ok=True)
    rows=[];sources=[]
    for m in chosen:
        sid=m['source_pose_id'];path=Path(m['source_review']).parent/sid/'protocol.json'
        p=prior.read(path);prior.verify(p);frame=p['frame']
        original=Path(frame['source_world']);base=ET.parse(original).getroot()
        names={v['object_id'] for v in frame['instance_mapping'].values()}
        ref=path.parent/'replay'/sid/'attempt-01/receipt.json';r=prior.read(ref);prior.verify(r)
        mask=ref.parent/'first-stable-window/frame-1-mask.bin'
        if r['status']!='original_pixel_evidence_certified' or any(x['missing_targets'] for x in r['full_mask_coverage']):
            raise ValueError('Source evidence incomplete')
        paths += [path,original,ref,mask]
        sources.append(dict(source_pose_id=sid,target_object_id=m['planned_object_id'],category=m['planned_category'],
            pair_id=m['pair_id'],source_frame=frame,reference_mask=str(mask),source_receipt=str(ref),
            existing_original_warm_cool=[x['member_id'] for x in candidates['members'] if x['pair_id']==m['pair_id']]))
        for condition in CONDITIONS:
            tree,changes=derive(base,m['planned_object_id'],names,condition)
            dest=worlds/f'{sid}-{condition}.sdf'
            if dest.exists():
                if signature(ET.parse(dest).getroot())!=signature(tree):raise ValueError('Frozen draft drift')
            else: ET.ElementTree(tree).write(dest,encoding='utf-8',xml_declaration=True)
            rows.append(dict(source_pose_id=sid,condition=condition,world_path=str(dest),changed_fields=changes,
                             semantic_world_identity=audit.fit.object_sha256(signature(tree)) if hasattr(audit.fit,'object_sha256') else None))
            paths.append(dest)
    # Same-component conditions can be genuine no-ops for assets with no front panel.
    from src.ml.artifacts import object_sha256
    for row in rows: row['semantic_world_identity']=object_sha256(signature(ET.parse(row['world_path']).getroot()))
    helpers=set()
    for m in chosen:
        cap=Path(m['variant_review']).parent/'protocol.json';c=prior.read(cap);prior.verify(c)
        helpers.add(c['helper']);paths += [cap,Path(c['helper'])]
    if len(helpers)!=1:raise ValueError('Ambiguous replay helper')
    models={}
    for dose,folder in [('low',audit.low.OUT),('high',audit.fit.OUT)]:
        pp=folder/'protocol.json';p=prior.read(pp);prior.verify(p);paths.append(pp)
        for key in audit.fit.KEYS:
            spec=p['models'][key];models[dose+'-'+key]=dict(weights=spec['weights'],weights_sha256=spec['weights_sha256'])
            paths.append(Path(spec['weights']))
    body=dict(status='diagnostic_drafts_frozen_not_captured',sources=sources,units=rows,models=models,
        helper=next(iter(helpers)),pilot_source_ids=['G01','G02','G03','G04'],
        pilot_selection='Previously frozen first four sources, one per class; not selected using model performance',
        expansion_rule='Only after all pilot source controls, mask/box/alignment checks and explicit full-label content reviews pass',
        contrasts=['palette: original/warm/cool versus gray_target_body at same pose',
                   'target panel: gray_target_full minus gray_target_body (report structural no-ops)',
                   'context bodies: gray_all_body minus gray_target_body',
                   'surrounding panels: all-full/all-body contrast minus target-full/target-body contrast'],
        inference=dict(device='cpu',imgsz=640,confidences=[.37,.001],iou=.7,max_det=300,agnostic_nms=False,matching_iou=.5),
        decisions=['Require changed-target and incidental-label metrics separately',
                   'A condition effect is diagnostic evidence, not proof a training augmentation will work',
                   'Changed switchgear training-fit remains limited; do not assume every class is already learned',
                   'Never delete unknown targets or select seeds; retain all conditions and seeds',
                   'No training-ready claim until full-label review, source isolation, exact exposure feasibility and real loader preflight'],
        training_ready=False,training_started=False,rendering_started=False,training_admitted=False,promotable=False,
        max_technical_attempts=3,source_role='same-source diagnostic, not newly independent training data',
        limits=['Shared layout/assets; not a sealed-scene generalization test','Preserve historical RGB, labels, weights and reviews',
                'Palette contrast changes RGB material values jointly, not a hue-only intervention'],
        inputs={str(p):prior.file_sha256(p) for p in paths})
    dest=OUT/'protocol.json'
    if dest.exists():
        old=prior.read(dest);prior.verify(old)
        if any(old[k]!=v for k,v in body.items()):raise ValueError('Protocol drift')
        return old
    return prior.frozen(dest,body)


if __name__=='__main__': print(freeze()['status'])
