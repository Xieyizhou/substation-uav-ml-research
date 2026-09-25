"""Explicit paired replacements for visually rejected coverage poses 34 and 41."""
import asyncio,copy,math,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.hard_negative_coverage import OUT,read,save,verify,file_sha256,candidates,prepare,evidence
from src.ml.artifacts import object_sha256
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.expansion import _pose
from src.vision.canonical.gates import validate_preflight
from src.vision.canonical.collect import collect

async def main():
    matrix=prepare();manifest=read(OUT/'remaining-review-manifest.json');verify(manifest)
    path=OUT/'repair-v2.json'
    if path.exists():spec=read(path);verify(spec)
    else:
        replacements=[];runs=[]
        for ordinal,reference in ((34,30),(41,42)):
            ref=next(r for r in matrix['poses'] if r['ordinal']==reference)
            row=copy.deepcopy(ref)
            # New camera location, fixed before any model scoring. Re-aim at
            # observed mixed structure / retain demonstrated edge offset.
            row['camera_position']=[ref['camera_position'][0]+.9,ref['camera_position'][1]-.7,ref['camera_position'][2]+.2]
            row['position'],row['orientation']=_pose(row['camera_position'],ref['look_at'],ref['offset']+(5 if ordinal==41 else 0))
            row.update(ordinal=ordinal,replaces_ordinal=ordinal,source_proposal_ordinal=reference,stage='repair',
                       family=f'{ref["map_id"]}:coverage-repair-v2:{ordinal}')
            row['pose_id']=object_sha256({'repair_v2':ordinal,**row})
            replacements.append(row)
            for light in ('light_cool_low','light_normal'):
                run=next(r for r in matrix['runs'] if r['stage']=='remaining' and f'-{ref["map_id"]}-{light}' in r['run_id'])
                src=Path(run['plan_path']);plan=read_record(src)
                v=copy.deepcopy(row);v.update(pair_id=row['pose_id'],variant=light,lighting_id=light,
                    view_id=object_sha256({'repair_v2':row['pose_id'],'variant':light}),derivation_group=f'coverage-v2:{row["pose_id"]}')
                folder=OUT/'runs'/f'repair-{ordinal}-{light}'/'plan';folder.mkdir(parents=True,exist_ok=False)
                for name in plan['files']:shutil.copyfile(src.parent/name,folder/name)
                record={k:copy.deepcopy(vv) for k,vv in plan.items() if k!='identity'};record.update(calibration_views=[v],parent_plan_identity=plan['identity'])
                written=write_record(folder/'plan.json',record);validate_preflight(written,folder,[v])
                runs.append(dict(plan_path=str(folder/'plan.json'),ordinal=ordinal,run_id=folder.parent.name))
        spec=save(path,dict(status='frozen_replacement_pending_capture',replacements=replacements,runs=runs,
            rejected_view_ids=[r['view_id'] for r in manifest['frames'] if r['ordinal'] in (34,41)],
            reason_by_ordinal={'34':'Only building visible, other projected structures occluded','41':'Actual cabinet is not truncated or occluded'},
            inputs={str(OUT/'matrix.json'):file_sha256(OUT/'matrix.json'),str(OUT/'remaining-review-manifest.json'):file_sha256(OUT/'remaining-review-manifest.json'),str(Path(__file__)):file_sha256(Path(__file__))}))
    completed=[]
    for run in spec['runs']:
        root=Path(run['plan_path']).parent.parent
        for attempt in range(1,4):
            folder=root/f'capture-attempt-{attempt:03}'
            receipt=read_record(folder/'collection-receipt.json') if (folder/'collection-receipt.json').exists() else await collect(run['plan_path'],folder)
            if any(r['status']=='rejected' for r in receipt['views']):raise ValueError('Semantic rejection')
            if receipt['status']=='complete_pending_review':break
        else:raise ValueError('Bounded retries exhausted')
        completed.append(dict(run_id=run['run_id'],receipt_path=str(folder/'collection-receipt.json'),inputs={str(folder/'collection-receipt.json'):file_sha256(folder/'collection-receipt.json')}))
        # evidence() verifies identities on each nested completion.
        completed[-1]=save(root/'capture-completion.json',completed[-1])
    save(OUT/'repair-capture.json',dict(status='complete_pending_review',runs=completed,inputs={str(path):file_sha256(path)}))
    evidence('repair')

if __name__=='__main__':asyncio.run(main())
