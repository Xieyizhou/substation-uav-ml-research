"""Strict read-only validation of all six real-loader receipts."""
import re
from scripts.vision.freeze_compensated_material_sequences import OUT,prior
from scripts.vision.brightness_transfer_runtime import check_log


def validate(p,r,seed):
    keys=[f'V-{seed}',f'VM-{seed}']
    if r['status']!='paired_actual_loaders_verified' or r['seed']!=seed or set(r['cells'])!=set(keys):raise ValueError('Cell coverage')
    if any(r[x] is not False for x in ('optimizer_created','backward_executed','training_validation_run')):raise ValueError('Forbidden training activity')
    for key in keys:
        c=r['cells'][key];seq=p['schedules'][key]
        if len(seq)!=2700 or c['actual']!=seq:raise ValueError('Actual exposure mismatch')
        check_log(p,key,c['brightness_log'])
        if len(c['batch_records'])!=450:raise ValueError('Incomplete batches')
        for i,b in enumerate(c['batch_records']):
            if b['step']!=i or b['members']!=seq[i*6:(i+1)*6]:raise ValueError('Batch member order')
            if set(b['full_supervision'])!={'cls','bboxes','batch_idx'}:raise ValueError('Incomplete full supervision')
            if any(not re.fullmatch('[0-9a-f]{64}',h) for h in [b['image_tensor_sha256'],*b['full_supervision'].values()]):raise ValueError('Invalid tensor identity')
    a,b=(r['cells'][k] for k in keys)
    for x,y in zip(a['batch_records'],b['batch_records'],strict=True):
        if x['full_supervision']!=y['full_supervision']:raise ValueError('Paired supervision conflict')
        if x['members']==y['members'] and x['image_tensor_sha256']!=y['image_tensor_sha256']:raise ValueError('Untreated tensor conflict')
    for i,(x,y) in enumerate(zip(a['brightness_log'],b['brightness_log'],strict=True)):
        if x['gain']!=y['gain']:raise ValueError('Brightness factor differs')
        if p['schedules'][keys[0]][i]==p['schedules'][keys[1]][i] and (x['before']!=y['before'] or x['after']!=y['after']):raise ValueError('Untreated brightness tensor differs')


def main():
    pp=OUT/'protocol.json';p=prior.read(pp);prior.verify(p)
    cp=OUT/'loader-completion.json';prior.verify(prior.read(cp));paths=[pp,cp]
    for row in p['pool_rows']:
        for kind in ('image','label'):
            if prior.file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Stale member file')
    for seed in (7,17,27):
        files=list((OUT/'loader-checks'/f'seed-{seed}').glob('attempt-*/complete.json'))
        if len(files)!=1:raise ValueError('Missing/duplicate receipt')
        r=prior.read(files[0]);prior.verify(r);validate(p,r,seed);paths.append(files[0])
    print('SIX_RECEIPTS_VALID; 16200_ACTUAL_EXPOSURES; NO_TRAINING')
    return paths


if __name__=='__main__':main()
