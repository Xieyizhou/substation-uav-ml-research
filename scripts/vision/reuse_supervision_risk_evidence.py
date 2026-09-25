"""Reanalyze a hash-valid same-frame mask; no renderer or capture."""
import argparse
from pathlib import Path
import shutil
from scripts.vision.supervision_risk_revision import OUT,read,verify,verify_tree,frozen,file_sha256,replay
from scripts.vision.instance_visibility_diagnosis import validate_world
import xml.etree.ElementTree as ET

def reuse():
    p=read(OUT/'protocol.json');verify(p);f=next(f for f in p['frames'] if f['event_id']=='T32')
    candidates=f['existing_replay_candidates']
    if len(candidates)!=1:raise ValueError('Nonunique reusable replay')
    rp=Path(candidates[0]);verify_tree(rp);old=read(rp)
    if old['status']!='original_pixel_evidence_certified' or not old['process_cleanup_complete'] or old['member_id']!=f['member_id']:
        raise ValueError('Reusable replay identity/status mismatch')
    validate_world(ET.parse(f['source_world']),ET.parse(rp.parent/'world.sdf'))
    dest=OUT/'reused-evidence/T32'
    if (dest/'receipt.json').exists():verify_tree(dest/'receipt.json');return
    dest.mkdir(parents=True,exist_ok=False)
    src=rp.parent/'first-stable-window';inputs={str(rp):file_sha256(rp),str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),str(Path(__file__)):file_sha256(Path(__file__))}
    for path in src.glob('frame-*'):
        if path.suffix in ('.json','.bin'):shutil.copyfile(path,dest/path.name);inputs[str(path)]=file_sha256(path)
    result=replay.analyze_selected(dest,f,old['move_fence_timestamp'])
    if result['status']!='original_pixel_evidence_certified':raise ValueError('Reused same-frame mask failed revalidation')
    inputs.update({str(x):file_sha256(x) for x in dest.iterdir() if x.is_file()})
    frozen(dest/'receipt.json',dict(**result,source_receipt=str(rp),event_id='T32',renderer_started=False,component_identity='unknown',inputs=inputs))
    print('REUSED_SAME_FRAME_T32',result['records'][0]['targets'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reuse',action='store_true');a=ap.parse_args()
    if a.reuse:reuse()
    else:print('PREFLIGHT_ONLY_NO_CAPTURE_NO_TRAINING')
