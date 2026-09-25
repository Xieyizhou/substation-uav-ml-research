"""Explicit AI-assisted observations of all H error boxes, never inferred from theme."""
import sys
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.inspect_anchor_errors import OUT,read,save,file_sha256,verify_tree
OBS={
1:[('building','灰色建筑正侧面、暗面板和底座。'),('cabinet','左侧边缘蓝色普通柜体平整侧面及底座。')],
2:[('cabinet','蓝色普通柜体侧面和黑色基座，立杆在框外。')],
3:[('cabinet','蓝色普通柜体正侧面、顶部及基座。')],
4:[('building','画面右侧截断灰色建筑大平面及底座，不见电容器阵列。'),('building','同一截断灰色建筑平面，另一个seed判为变压器。')],
5:[('cabinet','低冷光下蓝色柜体正面灰色面板、侧面及基座。')],
6:[('cabinet','常规光下同一蓝色柜体和矩形面板。')],
7:[('wall','框位于左图边缘，内容为低矮深色围墙端部及少量地面，不是画面中央杆体。')],
8:[('building','低冷光下灰色建筑侧面及右侧暗面板，右下有少量蓝色遮挡。')],
9:[('building','灰色建筑侧面及暗面板，被判为电容器组。'),('building','同一灰色建筑及底座，未见电容器结构。'),('building','同一灰色建筑侧面，置信度很高但内容仍非电容器。'),('cabinet','蓝色普通柜体平整背侧面及底座。')],
10:[('cabinet','蓝色普通柜体灰色前面板和侧面。'),('cabinet','同一蓝柜和前面板，另一seed也判开关柜。')],
11:[('cabinet','低冷光下蓝柜正侧面及深色矩形面板。')],
12:[('cabinet','常规光下蓝柜正侧面及灰色前面板。'),('cabinet','同一蓝柜，较高置信度开关柜误检。')],
13:[('cabinet','低冷光下蓝柜的大侧面及右侧小面板。')],
14:[('cabinet','蓝柜侧面、右侧灰色面板及基座。'),('mixed_structure','框跨灰色建筑立面、前方立杆及蓝柜一角，包含多个显著结构。'),('cabinet','蓝柜侧面、面板及基座，另一seed也误检。')],
15:[('building','低冷光下远处灰色建筑正面及大暗面板。'),('cabinet','左侧蓝色普通柜体正面灰面板及侧面。')],
16:[('building','灰色建筑正面和矩形暗面板，不见电容器阵列。'),('building','同一建筑正面，被第二个seed判为电容器组。'),('building','同一建筑暗面板，被第三个seed高置信度误检。'),('cabinet','左侧蓝柜灰色前面板及侧面，此框不是后方建筑。')],
17:[('building','右边界截断的灰色建筑立面和面板。'),('cabinet','蓝柜大侧面、部分前面板和基座。'),('cabinet','同一蓝柜侧面和顶部，另一个seed高置信度误检。'),('building','右图边缘截断灰色建筑面板及下缘。')],
18:[('cabinet','低冷光下蓝色普通柜体背侧平面及底座。')],
}

def validate(manifest,decisions):
    expected={p['prediction_id']:(f,p) for f in manifest['frames'] for p in f['predictions']}
    if len(expected)!=34 or len(decisions)!=34 or {d['prediction_id'] for d in decisions}!=set(expected):raise ValueError('Missing/duplicate decisions')
    for d in decisions:
        f,p=expected[d['prediction_id']]
        if d['decision']!='reviewed' or d['content_category']=='unknown' or not d['reason'] or d['review_nature']!='AI-assisted':raise ValueError('Unresolved review')
        for kind in ('image','evidence'):
            if d[kind+'_sha256']!=f[kind+'_sha256'] or file_sha256(f[kind+'_path'])!=d[kind+'_sha256']:raise ValueError('Stale review evidence')
        if any(d[k]!=p[k] for k in ('bbox_xyxy','confidence','seed','class_name')):raise ValueError('Prediction changed')

def main():
    path=OUT/'manifest.json';verify_tree(path);m=read(path);decisions=[]
    for f in m['frames']:
        obs=OBS[f['ordinal']]
        if len(obs)!=len(f['predictions']):raise ValueError('Explicit observation missing')
        for p,(category,reason) in zip(f['predictions'],obs):
            decisions.append(dict(**p,frame_ordinal=f['ordinal'],view_id=f['view_id'],variant=f['variant'],image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],
                content_category=category,reason=reason,decision='reviewed',review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat()))
    validate(m,decisions)
    r=save(OUT/'visual-review.json',dict(status='reviewed',decisions=decisions,counts=dict(Counter(d['content_category'] for d in decisions)),
        by_seed={str(s):dict(Counter(d['content_category'] for d in decisions if d['seed']==s)) for s in (7,17,27)},
        transitions={a:{str(s):dict(Counter(f'{int(r[a])}->{int(r["H"])}' for r in m['frame_transitions'] if r['seed']==s)) for s in (7,17,27)} for a in ('O','N')},
        inputs={str(p):file_sha256(p) for p in [path,Path(__file__)]+[Path(f[k+'_path']) for f in m['frames'] for k in ('image','evidence')]}))
    print({k:r[k] for k in ('counts','by_seed','transitions')})
    for key,curve in m['loss_curves'].items():
        print(key,{k:(curve[0][k],curve[-1][k]) for k in ('train/box_loss','train/cls_loss','train/dfl_loss')})

if __name__=='__main__':main()
