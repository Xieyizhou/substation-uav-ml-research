"""Explicit AI observations of the three rendered Q-300 prediction-crop pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.analyze_visibility_quality_results import OUT, read, save, file_sha256, verify_tree

OBS = {
 'F01':('cabinet_like','灰色长方体侧面及底座，无可辨识电抗器线圈。'),
 'F02':('cabinet_like','与 F01 同图的灰色长方体侧面、顶面和底座；不能仅凭该外形认定为变压器。'),
 'F03':('mixed_structure','框内大部分为网格地面，右侧有杆体下段；不是完整柜体。'),
 'F04':('mixed_structure','灰色箱体、前景黑色杆体及蓝色箱体边缘混合。'),
 'F05':('cabinet_like','灰色带深色面板结构的斜视局部，含底座及周围地面。'),
 'F06':('ground_background','主要是网格地面、远处低墙及天空，右边有细竖边；无可辨电容器组结构。'),
 'F07':('mixed_structure','框内主要是天空、低墙及左边杆体局部。'),
 'F08':('cabinet_like','正面矩形框架内有深色面板，含基座；仅能确定柜状混淆结构。'),
 'F09':('cabinet_like','与 F05 同图的面板结构侧缘及底座，其他 seed 再次误检。'),
 'F10':('ground_background','天空占大部，底部低墙转角和地面，右边细竖结构。'),
 'F11':('mixed_structure','天空、两根杆及低墙混合；与 F07 同图但预测框不同，不能当作同框复现。'),
 'F12':('mixed_structure','两侧杆体边段、天空和远处低墙，没有柜体主体。'),
 'F13':('mixed_structure','天空、低墙及左侧杆体局部；与 F03 同图但覆盖内容不同。'),
 'F14':('cabinet_like','蓝色柜状结构有深色前面板及基座，是明确的外形相似混淆。'),
 'F15':('mixed_structure','杆体、天空、低墙及底部蓝色箱体上沿混合。'),
 'F16':('mixed_structure','杆体、灰色箱体、蓝色箱体边缘、地面及天空混合，框未紧贴一个对象。'),
}

def main():
    path=OUT/'negative-review.json'
    if path.exists(): verify_tree(path); print('VERIFIED_EXISTING'); return
    verify_tree(OUT/'analysis.json'); r=read(OUT/'analysis.json')
    if set(OBS)!={x['item_id'] for x in r['negative_review_items']}: raise ValueError('Missing explicit observation')
    rows=[]
    for item in r['negative_review_items']:
        category,reason=OBS[item['item_id']]
        rows.append({**item,'review_status':'reviewed','review_nature':'AI辅助审核','reviewed_at':datetime.now(timezone.utc).isoformat(),'content_category':category,'reason':reason,'same_image_seeds':sorted({x['seed'] for x in r['negative_review_items'] if x['image_sha256']==item['image_sha256']})})
    save(path,dict(status='all_16_prediction_crops_reviewed',items=rows,unique_images=r['negative_unique_images'],scope='Predicted-box contents only; no inferred asset identity or label revision. Same image recurrence is not same-box agreement.',inputs={str(p):file_sha256(p) for p in (OUT/'analysis.json',Path(__file__))}))
    print('REVIEWED',len(rows),'UNIQUE',r['negative_unique_images'])

if __name__=='__main__':main()
