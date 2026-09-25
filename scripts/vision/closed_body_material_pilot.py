"""Four-source technical pilot for closed-body materials; never trains."""
import argparse,asyncio,copy,fcntl,os,shutil
from pathlib import Path
from scripts.vision.test_body_material_applicability import OUT as AUDIT,PRIOR,source_index,read,verify,frozen,file_sha256,ROOT
from scripts.vision.brightness_lr_retention import OUT as BASELINE
from scripts.vision.instance_visibility_diagnosis import add_sensor,validate_world,ET,raw_box
import scripts.vision.run_visibility_cleanup_validation as replay
from scripts.vision.run_body_visibility_intervention import analyze as diagnostic_analyze

OUT=ROOT/'data/research/ml_training_recovery_v1/closed-body-material-control-v1'
PALETTES={'warm':'0.55 0.48 0.36 1','cool':'0.25 0.28 0.32 1'}


def change_body(tree,name,color):
    out=copy.deepcopy(tree)
    matches=out.findall(f".//world/model[@name='{name}']/link/visual[@name='body']")
    if len(matches)!=1:raise ValueError('Missing/duplicate target body')
    body=matches[0]
    if body.find('material/pbr') is not None or body.find('material/script') is not None:raise ValueError('Unsupported material')
    for key in ('ambient','diffuse'):
        node=body.find('material/'+key)
        if node is None or len(node.text.split())!=4 or node.text.split()[-1]!='1':raise ValueError('Unexpected body alpha/material')
        node.text=color
    return out


def select():
    d=read(AUDIT/'diagnosis.json');e=read(PRIOR/'evidence.json');rv=read(PRIOR/'review.json')
    for r in (d,e,rv):verify(r)
    decisions={x['review_id']:x for x in rv['decisions']};evidence={x['review_id']:x for x in e['training']}
    sources,paths=source_index();chosen=[]
    for cls in ('capacitor_bank','switchgear'):
        seen=set()
        for r in sorted(d['training'],key=lambda x:x['member_id']):
            decision=decisions[r['review_id']]
            if r['class_name']!=cls or r['gaps'] or decision['decision']!='condition_recorded' or decision['occlusion_condition']!='no_obvious_foreground_occlusion' or min(r['actual_exposures'].values())<=0:continue
            s=sources[r['member_id']];lineage=s.get('derivation_group') or evidence[r['review_id']]['target']['lineage_id']
            if lineage in seen:continue
            seen.add(lineage);chosen.append(dict(r,lineage_id=lineage,evidence=evidence[r['review_id']]))
            if len(seen)==2:break
        if len(seen)!=2:raise ValueError('Two distinct registered pose groups per class unavailable')
    return chosen,paths


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():verify(read(dest));return read(dest)
    for p in (AUDIT/'completion.json',ROOT/'data/research/ml_training_recovery_v1/asset-visibility-replay-v1/completion.json',BASELINE/'completion.json'):verify(read(p))
    chosen,paths=select();OUT.mkdir(exist_ok=True);frames=[]
    for r in chosen:
        cp=Path(r['capture_path']);receipt=read(cp);vid=Path(r['source_image_path']).parent.name
        view=next(v for v in receipt['views'] if v['view_id']==vid);pp=Path(r['world_path']).parent/'plan.json';plan=read(pp)
        sp=OUT/(r['review_id']+'-source.json');frozen(sp,{**view,'inputs':{str(cp):file_sha256(cp)}})
        frames.append(dict(member_id=r['member_id'],lineage_id=r['lineage_id'],class_name=r['class_name'],review_ids=[r['review_id']],
            source_image=r['source_image_path'],source_receipt=str(sp),source_plan=str(pp),source_world=r['world_path'],
            actual_pose=view['actual_pose'],world_name=plan['world_name'],instance_mapping=receipt['collection_checks']['instance_mapping'],
            events=[dict(review_id=r['review_id'],runtime_label=int(r['runtime_label']),object_id=r['object_id'],bbox_xyxy=r['evidence']['target']['truth']['bbox_xyxy'])]))
        paths += [cp,sp,pp,pp.parent/'obstacles.json',Path(r['world_path']),Path(r['source_image_path'])]
    helper=replay.OUT/'gz_visibility_capture_cleanup_fixed';local=OUT/helper.name;shutil.copy2(helper,local)
    baseline=read(BASELINE/'protocol.json');verify(baseline)
    paths += [helper,local,Path(__file__),Path(replay.__file__),ROOT/'scripts/vision/run_body_visibility_intervention.py',
        AUDIT/'diagnosis.json',AUDIT/'completion.json',PRIOR/'evidence.json',PRIOR/'review.json',BASELINE/'protocol.json',BASELINE/'completion.json']
    return frozen(dest,dict(status='four_source_pilot_frozen_not_training_ready',frames=frames,palettes=PALETTES,
        geometry_policy='Closed body, base, internal cylinders, panels and all collisions unchanged. Change only selected equipment body ambient/diffuse RGB.',
        selection_rule='Two lexicographically first eligible members per class, distinct registered pose lineage; no model scores. Registered lineage is not independent asset/scene certification.',
        pilot_limit='4 original controls + 8 material variants; first three stable frames per attempt; maximum 3 technical attempts per unit; semantic/alignment failures stop that source, no replacement member.',
        future_training=dict(reference=str(BASELINE/'protocol.json'),learning_rate=.0005,seeds=[7,17,27],optimizer_steps=450,
            initialization=baseline['initialization'],evaluation=baseline['evaluation'],retention=baseline['retention'],acceptance_policy=baseline['acceptance_policy'],
            policy='Preserve reference image/class/lineage exposure, batches, negative slots and exact brightness gains. Final replacement sequence requires a separate fully reviewed dataset freeze; pilot does not authorize training.',
            inference_thresholds_unchanged=True),training_ready=False,training_started=False,
        inputs={str(p):file_sha256(p) for p in paths}))


def material_analysis(folder,frame,fence):
    result=diagnostic_analyze(folder,frame,fence)
    old=read(frame['source_receipt'])['raw_truth']['annotatedBox']
    for record in result['records']:
        new=record['boxes'].get('annotatedBox',[])
        if len(new)!=len(old):raise ValueError('Material changed full box membership')
        deltas=[]
        for b in old:
            matches=[x for x in new if x.get('label',0)==b.get('label',0)]
            if len(matches)!=1:raise ValueError('Missing/duplicate material instance')
            deltas.append(max(abs(a-c) for a,c in zip(raw_box(b),raw_box(matches[0]))))
        record['maximum_box_delta_px']=max(deltas,default=0)
        if record['maximum_box_delta_px']>1:raise ValueError('Material box drift exceeds one pixel')
    result.update(status='material_render_passed_full_label_review_pending',reason='Only body material changed; complete box membership and <=1px coordinates checked; no automatic semantic admission.')
    return result


async def run():
    p=freeze()
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[]
        for frame in p['frames']:
            control_ok=False
            for variant in ('control',*PALETTES):
                if variant!='control' and not control_ok:break
                unit=OUT/'renders'/frame['review_ids'][0]/variant;unit.mkdir(parents=True,exist_ok=True)
                up=unit/'protocol.json'
                if not up.exists():
                    shutil.copy2(OUT/'gz_visibility_capture_cleanup_fixed',unit/'gz_visibility_capture_cleanup_fixed')
                    frozen(up,dict(frame=frame,variant=variant,color=PALETTES.get(variant),inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),str(unit/'gz_visibility_capture_cleanup_fixed'):file_sha256(unit/'gz_visibility_capture_cleanup_fixed')}))
                else:verify(read(up))
                old=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze);oldlog=os.environ.get('GZ_LOG_PATH')
                replay.OUT=unit
                if variant!='control':
                    name=frame['events'][0]['object_id'];color=PALETTES[variant]
                    replay.add_sensor=lambda tree:add_sensor(change_body(tree,name,color))
                    replay.validate_world=lambda original,derived:validate_world(change_body(original,name,color),derived)
                    replay.analyze=material_analysis
                logs=unit/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs);last=None;rp=None
                try:
                    for n in range(1,4):
                        rp=unit/'replay'/frame['review_ids'][0]/f'attempt-{n:02}'/'receipt.json'
                        if rp.exists():last=read(rp);verify(last)
                        elif rp.parent.exists():continue
                        else:last=await replay.attempt(frame,n)
                        if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
                        if last['status']!='technical_failure':break
                finally:
                    replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze=old
                    if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
                    else:os.environ['GZ_LOG_PATH']=oldlog
                results.append(dict(review_id=frame['review_ids'][0],variant=variant,status=last['status'] if last else 'attempts_exhausted',receipt=str(rp) if last else None))
                if variant=='control':control_ok=bool(last and last['status']=='original_pixel_evidence_certified')
        dest=OUT/'render-summary.json'
        if dest.exists():verify(read(dest));return
        paths=[OUT/'protocol.json',*[Path(r['receipt']) for r in results if r['receipt']]]
        frozen(dest,dict(status='pilot_rendering_finished_review_pending',units=results,training_ready=False,training_started=False,
            inputs={str(x):file_sha256(x) for x in paths}))
        print([(r['review_id'],r['variant'],r['status']) for r in results],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');a=ap.parse_args()
    if a.render:asyncio.run(run())
    else:freeze();print('FROZEN_NO_TRAINING_NO_RENDERING')
