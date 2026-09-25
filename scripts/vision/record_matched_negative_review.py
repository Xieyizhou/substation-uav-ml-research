"""Explicit observations recorded only after viewing each prediction's evidence."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.run_matched_appearance_training import OUT,KEYS,read,save,file_sha256,verify_tree

OBS = {
    ('L-300-27','4822682fe90f97768dc7fde6823584c60f9bacb3f0cd34d9fcd49946f0730c56',0):
      ('柜体','近处蓝绿色长方体柜状结构的无面板侧／背面、顶面和黑色底座占据预测框；左侧粗杆基本在框外。全图与放大图均已查看，预测的主要内容不是杆体。'),
    ('L-300-27','93b84f7d6bd6b76bc7cb00414b8d2d55ca1cf4a5ca312a69d3cf3aa99efd825f',0):
      ('柜体','近处蓝绿色柜状主体、宽矩形深灰面板、顶面、侧面及底座清楚可见；背景灰色柜和杆不构成框内主体。已查看全图叠框及局部放大。'),
    ('L-300-27','932090be7b6e19bd1ebaa0de21b16707a18b25176ae3b8046e0a7688ad7d40cc',0):
      ('柜体','蓝绿色柜状正面带深灰矩形面板，右侧面、顶面及黑色基座可见；框含少量地面，未把右侧杆体作为主要内容。全图及放大证据均已查看。'),
    ('L-300-27','be27b3ad6515bebedabb497dd41fd5e0031452aaa76a0cc3c1da665ec69d52fc',0):
      ('柜体','中距离蓝绿色柜状结构以无面板的大侧面为主，右侧窄面上有深灰面板，底座可见；相邻粗杆在框外。全图叠框与框内放大均已查看。'),
    ('L-300-27','be31051ed49f795ddec22e17bae4e87b94d3b97c91d8935ea451b666c3c2618e',0):
      ('柜体','画面左下方蓝绿色柜状主体、矩形深色面板、右侧面与底座被框住。该图 K-300-27 的误检在另一个灰色柜上，不能把两个预测当成同一对象一致出现。已分别查看本预测全图及放大图。'),
    ('L-300-27','c9a88d83bad1b5155b9162481e7ca4a8e6d020e34246c3e4159dc74f4f91d99b',0):
      ('柜体','较远处蓝绿色柜状结构的无面板宽背／侧面和黑色底座占据框内；邻近杆体位于框外。即使面板不可见，仍可辨识连续长方体外观。全图与放大图均已查看。'),
    ('K-300-27','be31051ed49f795ddec22e17bae4e87b94d3b97c91d8935ea451b666c3c2618e',0):
      ('柜体','框内为灰色柜状主体正面、深色矩形面板和黑色底座，侧边含少量背景；主要内容不是相邻杆体。全图叠框及框内放大均已查看，外观被预测为电容器组。仅描述可见混淆结构，不认证设备实例身份。'),
    ('L-300-17','c91cd46a02d2f96cbf4ddbc8021d907059e506a8a346eb66640d200f8bea8401',0):
      ('柜体','框内主要为图像右缘被截断的灰色柜状结构、深色矩形面板及底座，另含地面和阴影；不是杆体。全图叠框与局部放大均已查看。此判断描述可见内容，不认证仿真实例身份。'),
}

def main():
    dest=OUT/'negative-review.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING',dest);return
    expected={};inputs={str(Path(__file__)):file_sha256(Path(__file__))};seen=set()
    for cell in KEYS:
        mp=OUT/'negative-evidence'/cell/'manifest.json';verify_tree(mp,seen);inputs[str(mp)]=file_sha256(mp)
        for frame in read(mp)['frames']:
            for box in frame['boxes']:
                key=(cell,frame['view_id'],box['index'])
                if key in expected:raise ValueError('Duplicate prediction identity')
                expected[key]=(frame,box)
    if set(expected)!=set(OBS):raise ValueError('Unreviewed or unexpected prediction; do not auto-pass')
    decisions=[]
    for key,(frame,box) in expected.items():
        category,reason=OBS[key]
        decisions.append(dict(cell=key[0],seed=int(key[0].split('-')[-1]),view_id=key[1],prediction_index=key[2],
            variant=frame['variant'],content_category=category,reason=reason,review_nature='AI辅助审核',
            reviewed_at=datetime.now(timezone.utc).isoformat(),status='observed_content_not_instance_certification',
            image_sha256=frame['image_sha256'],image_path=frame['image_path'],bbox_xyxy=box['bbox_xyxy'],
            predicted_class=box['class_name'],confidence=box['confidence'],crop_path=box['crop_path'],
            crop_sha256=file_sha256(box['crop_path']),overlay_path=frame['overlay_path'],overlay_sha256=file_sha256(frame['overlay_path'])))
    save(dest,dict(status='all_formal_negative_predictions_explicitly_reviewed',decisions=decisions,
        prediction_events=len(decisions),unique_images=len({x['image_sha256'] for x in decisions}),inputs=inputs,
        limits='Prediction events across seeds are not independent image samples; content descriptions do not establish unique causal roots.'))
    print('NEGATIVE_REVIEW',len(decisions),flush=True)

if __name__=='__main__':main()
