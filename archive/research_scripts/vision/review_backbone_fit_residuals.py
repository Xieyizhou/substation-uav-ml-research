"""Human-readable AI observations of all fifteen displayed residual pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_routed_backbone_fit import OUT
from scripts.vision.routed_backbone_control import checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
'G02-gray_all_body':{0:'后灰柜顶面、面板及侧面可见，下部被近柜遮挡。',5:'右近灰柜顶面、面板、侧面与完整基座清楚。',6:'左灰柜大顶面、侧背面与基座清楚，无面板可见，无主要遮挡。'},
'G04-gray_all_body':{1:'后灰圆柱顶面、大主体和左基座清楚，右下被近变压器遮挡。'},
'S02-cool':{2:'冷灰电容器顶面、两个主体面及完整基座清楚，未见主要遮挡或截断。'},
'S02-gray_all_body':{1:'左灰变压器三柱、顶面、两个主体面及完整基座清楚。',2:'灰电容器顶面、两个主体面及完整基座清楚，未见主要遮挡。'},
'S02-gray_target_body':{2:'目标灰电容器顶面、两个主体面及完整基座清楚，前景变压器保持青色，不是另一个独立位姿。'},
'S02-warm':{2:'暖色电容器顶面、两个主体面及完整基座清楚，无主要遮挡，属于S02同源变体。'},
'S03-warm':{1:'暖色柜体大顶面、端面和基座可见，高俯视、无正面面板可见；不是仅底座。'},
'S04-gray_all_body':{0:'右后灰柜顶面、侧面和面板局部可见，下部被前柜遮挡。',5:'右近灰柜大顶面、侧面、左侧面板及基座清楚。',6:'左灰柜顶面、两个侧背面和基座清楚，无主要遮挡，无正面面板。'},
'S07-gray_all_body':{1:'左后圆柱顶面、大主体和基座可见，右下极小部分被前变压器遮挡。'},
'S08-gray_all_body':{1:'后灰电容器顶面、大主体、右侧面及基座清楚，未见主要遮挡。'},
'small-material-v1-S02':{0:'中柜顶面、前面板轮廓、侧面和右基座可见，左下被近柜遮挡。',1:'右后变压器三柱、顶面、两个主体面和基座可见，右侧有前景粗杆遮挡。',2:'左变压器三柱、顶面、大主体及基座清楚。',4:'后电容器顶面、大主体和右基座可见，左下被前变压器遮挡。'},
'small-material-v1-S04':{0:'冷灰中柜顶面、面板轮廓、侧面与右基座可见，左下被近柜遮挡。',1:'冷灰右后变压器三柱、顶面、两个主体面和基座可见，右部被粗杆遮挡。',2:'冷灰左变压器三柱、顶面、大主体和基座清楚。',3:'近柜顶面、前面板轮廓、侧面与完整基座清楚。',4:'冷灰后电容器顶面、大主体与右基座可见，左下被前变压器遮挡。'},
'small-material-v1-S10':{0:'后柜顶面和主体上部带可见，下部被前柜遮挡；框内前景顶面不能归给后柜。'},
'small-material-v1-S14':{3:'后变压器三柱、顶面及主体上部可见，下部被近变压器遮挡；前景三柱不属于后目标。',4:'右变压器三柱、顶面、大主体和右基座可见，左下被近柜遮挡。',5:'远柜顶面、侧背面及左基座可见，右部被变压器遮挡，尺度小。',6:'远电容器顶面、两个主体面与基座清楚，无主要遮挡。'},
'small-material-v1-S16':{0:'右近冷灰柜顶面、大侧背面与基座清楚，无正面面板可见。',2:'近冷灰变压器三柱、大顶面、主体及基座清楚。',3:'后冷灰变压器三柱、顶面及主体上部可见，下部被近变压器遮挡。',5:'远冷灰柜顶面、侧背面和左基座可见，右部被变压器遮挡，尺度小。',6:'远冷灰电容器顶面、两个主体面与基座清楚，无主要遮挡。'},
}

def validate(e,decisions):
    expected={(f['member_id'],x['key'],x['truth_index']): (f,x) for f in e['frames'] for x in f['events']}
    if len(decisions)!=len(expected):raise ValueError('Missing or duplicate decisions')
    seen=set()
    for d in decisions:
        k=(d['member_id'],d['event']['key'],d['event']['truth_index'])
        if k in seen or k not in expected:raise ValueError('Duplicate or unknown decision')
        seen.add(k);f,x=expected[k]
        if d['event']!=x or not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Changed event or missing reason')
        for field,source in [('image_sha256','image_sha256'),('label_sha256','label_sha256'),('evidence_sha256','page_sha256')]:
            if d[field]!=f[source]:raise ValueError('Stale review identity')
        if file_sha256(f['image'])!=f['image_sha256'] or file_sha256(f['page'])!=f['page_sha256']:raise ValueError('Stale evidence')

def run():
    ep=OUT/'residual-review-v1/evidence.json';e=checked(ep)
    if set(NOTES)!={f['member_id'] for f in e['frames']}:raise ValueError('Changed image coverage')
    decisions=[];stamp=datetime.now(timezone.utc).isoformat()
    for f in e['frames']:
        if set(NOTES[f['member_id']])!={x['truth_index'] for x in f['events']}:raise ValueError('Changed target coverage')
        for x in f['events']:
            decisions.append(dict(member_id=f['member_id'],event=x,reason=NOTES[f['member_id']][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],label_sha256=f['label_sha256'],evidence_sha256=f['page_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    validate(e,decisions)
    return write_record(ep.parent/'review.json',dict(status='all_residual_misses_visually_reviewed',unique_images=len(e['frames']),events=len(decisions),decisions=decisions,limits='Fitting diagnostic only. Seeds and same-source variants do not add independent scenes. No pixel visibility certification or label edits.',training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,ep.parent/'build-receipt.json',Path(__file__))}))

if __name__=='__main__':print(run()['events'])
