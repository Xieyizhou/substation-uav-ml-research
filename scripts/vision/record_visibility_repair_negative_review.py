"""Explicit observations after viewing all six full-frame and crop panels."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.audit_visibility_repair_results import OUT, read, save, file_sha256

OBS = {
 'F01': ('柜体', '蓝绿色长方体柜体、深灰正面面板及黑色基座；框内主要内容不是后方杆体。'),
 'F02': ('柜体', '蓝绿色柜体正面深灰面板、右侧面及底座；右侧粗杆在框外。'),
 'F03': ('混合结构', '灰色柜体被前景粗杆遮挡，框右下还包含蓝绿色柜体局部；无法把预测归因于单一组件。'),
 'F04': ('地面／阴影', '框内主要为网格地面和杆影，左缘包含杆体底部；不是前景蓝绿色柜体。'),
 'F05': ('地面／阴影', '框内主要是网格地面与细长阴影，边缘少量杆体及远处基座；未见完整柜状主体。'),
 'F06': ('柜体', '右缘截断的灰色柜体、深色面板和底座，另含地面背景；此记录不认证实例身份。'),
}

def main():
    dest = OUT/'negative-review.json'
    if dest.exists(): raise ValueError('Do not overwrite review')
    manifest = OUT/'evidence.json'; data = read(manifest)
    if len(data['events']) != len(OBS) or {e['event_id'] for e in data['events']} != set(OBS):
        raise ValueError('Missing or duplicate decisions')
    inputs = {str(manifest):file_sha256(manifest), str(Path(__file__)):file_sha256(Path(__file__))}
    decisions = []
    for e in data['events']:
        for kind in ('image','evidence'):
            if file_sha256(e[kind+'_path']) != e[kind+'_sha256']: raise ValueError('Stale evidence')
            inputs[e[kind+'_path']] = e[kind+'_sha256']
        category, reason = OBS[e['event_id']]
        decisions.append({**e,'content_category':category,'reason':reason,'review_nature':'AI辅助审核',
            'reviewed_at':datetime.now(timezone.utc).isoformat(),'status':'observed_content_not_instance_certification'})
    save(dest,dict(status='six_negative_prediction_events_reviewed',decisions=decisions,
        unique_images=len({e['image_sha256'] for e in decisions}),inputs=inputs,
        remaining='Positive-frame regression visual audit, deep artifact verification and next-design preflight remain pending.'))
    print(dest)

if __name__ == '__main__': main()
