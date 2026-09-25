"""Independent correction of diagnostic truth adapter; preserves failed v1."""
import argparse
from pathlib import Path
from scripts.vision import infer_transfer_pilot as base

OLD=base.OUT
OUT=OLD.with_name('inference-v2')


def freeze():
    base.review(); prior=base.prior
    dest=OUT/'protocol.json'
    if dest.exists():
        p=prior.read(dest);prior.verify(p);return p
    oldpath=OLD/'protocol.json'; old=prior.read(oldpath);prior.verify(old)
    cp=base.CAND/'reviewed-completion.json';c=prior.read(cp);prior.verify(c)
    members={m['member_id']:m for m in c['members']}
    for row in old['members']:
        if row['condition'] in ('warm','cool'):
            m=members[row['member_id']]
            resolved=base.resolve_truth(m['full_truth'],m['instance_mapping'])
            row['truth']=[dict(t,object_id=r['object_id']) for t,r in zip(m['full_truth']['objects'],resolved)]
        for t in row['truth']:
            if not t.get('class_name') or not t.get('object_id') or len(t['bbox_xyxy'])!=4:
                raise ValueError('Incomplete truth adapter')
    paths=[oldpath,cp,Path(__file__).resolve()]
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(**{k:v for k,v in old.items() if k not in ('identity','inputs')},
        correction='v1 resolved category lacked scorer class_name; restore full original truth with resolved object ID. No labels changed.',
        inputs={**old['inputs'],**{str(x):prior.file_sha256(x) for x in paths}}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args()
    p=freeze();base.OUT=OUT
    if a.infer:
        for key in p['models']:base.infer(key,p)
    else:print('PREFLIGHT_ONLY',len(p['members']))


if __name__=='__main__':main()
