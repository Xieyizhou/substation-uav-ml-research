"""Explicit crop observations and audited training-coverage crosswalk."""
import sys
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.inspect_budget_errors import OUT,RUN,BASE,KEY,read,save,file_sha256,verify_tree
OBS={
'N01':('cabinet','蓝色普通柜体正面暗面板、侧面及基座。'),
'N02':('building','灰色建筑正侧立面、大暗面板及基座。'),
'N03':('building','右缘截断的灰色建筑大平面及基座。'),
'N04':('building','低冷光下右缘截断的建筑暗平面和底座。'),
'N05':('building','灰色建筑侧面及右侧暗面板，框边有少量地面。'),
'N06':('building','与同帧另一框相同的建筑立面及暗面板。'),
'N07':('building','建筑大平面及右侧暗面板；右框缘少量立杆，不是框主体。'),
'N08':('building','灰色建筑正面的大暗矩形面板及底座。'),
'N09':('building','低冷光下建筑正面与大暗矩形面板。'),
'N10':('mixed_structure','框跨建筑立面、前景黑色立杆及蓝色柜体。'),
'N11':('mixed_structure','框内黑色立杆、蓝柜及后方灰色建筑均显著。'),
'N12':('mixed_structure','灰色建筑立面与前方蓝柜、黑色立杆共同位于框内。'),
'N13':('building','灰色建筑正面及横向纹理暗面板。'),
'N14':('building','同一建筑正面和暗面板，被另一类别重复预测。'),
'N15':('building','低冷光下建筑正面的大矩形暗面板。'),
'N16':('building','灰色建筑大侧面、右侧暗面板及底座。'),
'N17':('building','低冷光下灰色建筑侧面与右侧暗面板。'),
'N18':('building','图像右缘截断的灰色建筑边缘、暗面板和地面。'),
'N19':('building','灰色建筑大侧面及基座；右缘窄条立杆。'),
'L01':('target_visible','较大的暗蓝变压器主体、顶部套管和底座可见，部分顶部结构被裁剪边界截断；不是纯小目标。'),
'L02':('target_partly_occluded','后排开关柜正面暗面板和蓝侧面可见，前方柜体及圆柱遮挡下部。'),
'L03':('target_partly_occluded','后排蓝柜正侧面可见，前方蓝柜遮住右下部。'),
'L04':('target_visible','较大的蓝柜顶部与无面板暗侧面，呈大面积简单平面。'),
'L05':('target_partly_occluded','后排柜体顶部和窄正面可见，前排柜体遮挡大部分下部。'),
'L06':('target_visible','较大的蓝柜完整顶部、暗色侧面及基座，没有正面面板细节。'),
'L07':('target_partly_occluded','后排蓝柜顶部与窄侧面，前方蓝柜遮挡其下部。'),
'L08':('target_visible','蓝柜大侧面、顶部和基座可见，缺少正面面板特征。'),
}

def validate(manifest,decisions):
    expected={x['item_id']:x for x in manifest['items']}
    if len(decisions)!=len(expected) or {d['item_id'] for d in decisions}!=set(expected):raise ValueError('Missing or duplicate review')
    for d in decisions:
        item=expected[d['item_id']]
        if not d['reason'] or d['review_nature']!='AI-assisted' or d['decision']!='reviewed':raise ValueError('Unresolved review')
        if d['bbox_xyxy']!=item['bbox_xyxy']:raise ValueError('Changed box')
        for p,h in ((item['source']['image_path'],d['image_sha256']),(item['evidence_path'],d['evidence_sha256'])):
            if file_sha256(p)!=h:raise ValueError('Stale evidence')

def main():
    mp=OUT/'manifest.json';verify_tree(mp);m=read(mp);decisions=[]
    for item in m['items']:
        category,reason=OBS[item['item_id']]
        decisions.append(dict(item_id=item['item_id'],bbox_xyxy=item['bbox_xyxy'],image_sha256=item['source']['image_sha256'],evidence_sha256=item['evidence_sha256'],
            content_category=category,reason=reason,decision='reviewed',review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat()))
    validate(m,decisions)
    admission_path=BASE/'hard-negative-coverage-v1/final-admission.json';verify_tree(admission_path)
    admission=read(admission_path);frames={r['image_path']:r for r in admission['frames']};reviews={r['view_id']:r for r in admission['decisions']}
    p=read(RUN/'protocol.json');counts=Counter(p['schedules'][KEY]);coverage=[];weighted=Counter();examples=[]
    for row in p['pool_rows']:
        if not row.get('coverage_unit'):continue
        f=frames[row['source_image_path']];review=reviews[f['view_id']]
        if review['decision']!='accepted' or review['image_sha256']!=f['image_sha256']:raise ValueError('Invalid source review')
        contents=sorted({roi['content'] for roi in review['rois']});n=counts[row['member_id']]
        for content in contents:weighted[content]+=n
        coverage.append(dict(member_id=row['member_id'],unit=row['coverage_unit'],actual_exposures=n,reviewed_contents=contents,source_review=review))
        if n and row['coverage_unit'] in ('B1','B2','M1') and len([x for x in examples if x['coverage_unit']==row['coverage_unit']])<2:examples.append(row)
    canvas=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(examples):
        assert file_sha256(row['image_path'])==row['image_sha256']
        im=Image.open(row['image_path']).convert('RGB');im.thumbnail((590,260));x=i%2*600;y=i//2*300;canvas.paste(im,(x,y+35));draw.text((x,y+5),f"{row['coverage_unit']} actual exposures={counts[row['member_id']]}",fill='black')
    example_path=OUT/'training-examples.jpg';canvas.save(example_path)
    save(OUT/'review-and-coverage.json',dict(status='reviewed',decisions=decisions,negative_content_counts=dict(Counter(d['content_category'] for d in decisions if d['item_id'].startswith('N'))),
        coverage=coverage,new_negative_content_image_exposures=dict(weighted),coverage_count_semantics='An exposure can contain several content types; not independent instance supervision. Old negatives excluded from this content crosswalk.',
        training_examples=examples,inputs={str(q):file_sha256(q) for q in (mp,admission_path,Path(__file__),example_path)}))
    print('COUNTS',Counter(d['content_category'] for d in decisions));print('COVERAGE',weighted)

if __name__=='__main__':main()
