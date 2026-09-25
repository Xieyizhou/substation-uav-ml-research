"""Four explicit pilot observations; expansion gate, not full training approval."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.physical_low_light_capture import OUT,freeze,prior
NOTES={
'L03':{'capacitor_east':'宽箱体顶面、深色立面与基座均清楚；降低光照后立面仍可区分，没有新增遮挡或图缘截断。'},
'L07':{'reactor_north':'圆柱椭圆顶面、侧壁与基座均可辨，暗侧轮廓保留；未见前景遮挡或截断。'},
'L11':{'west_switchgear_02':'后方窄柜顶面、面板边框与侧边清楚，底部附近有前景柜顶面；主要面板仍可辨，不称完全无遮挡。','switchgear_west':'后方宽柜面板、横纹、侧面和基座可见，无图缘截断。','entry_switchgear':'近柜面板边框、顶面、侧边和底座清楚，暗面内横纹仍可分。'},
'L15':{'west_switchgear_03':'右侧斜俯视柜顶面、前面板及侧面清楚，未见图缘截断或主要遮挡。','transformer_mid':'俯视箱体三根顶部突出部、宽顶面、窄立面和基座清楚；侧向几何覆盖仍有限。'},
}

def main():
    p=freeze();dest=OUT/'pilot-quality.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    if set(NOTES)!=set(p['pilot_units']):raise ValueError('Pilot inventory mismatch')
    deps=[OUT/'capture-protocol.json',Path(__file__).resolve()];ds=[]
    for uid,notes in NOTES.items():
        ep=OUT/'evidence'/uid/'evidence.json';rp=OUT/'replays'/uid/'completion.json'
        e,r=prior.read(ep),prior.read(rp);prior.verify(e);prior.verify(r);deps.extend([ep,rp])
        if r['status']!='low_light_capture_exact_replay_verified' or set(notes)!={x['object_id'] for x in e['events']}:raise ValueError('Incomplete pilot evidence')
        for x in e['events']:
            ds.append(dict(event_id=x['event_id'],reason=notes[x['object_id']],review_nature='AI辅助审核',review_time=datetime.now(timezone.utc).isoformat(),evidence_identity=e['identity'],crop_sha256=x['crop_sha256'],status='content_sufficient_for_bounded_research',training_admitted=False,promotable=False))
    return prior.frozen(dest,dict(status='pilot_explicitly_reviewed_expand_frozen_32',decisions=ds,full_frames_viewed=list(NOTES),
        scope='All seven labels and four full frames visually inspected; exact replay supplies full mapped-instance coverage. Only these four pilot frames approved; remaining 28 require independent review.',
        inputs={str(p):prior.file_sha256(p) for p in deps}))

if __name__=='__main__':print(main()['status'])
