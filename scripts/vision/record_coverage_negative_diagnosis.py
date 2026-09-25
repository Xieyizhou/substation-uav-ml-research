"""Explicit box-content decisions after viewing all fifteen evidence images."""
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.inspect_coverage_negative_errors import OUT,read,save,file_sha256,verify_tree

OBSERVATIONS={
1:[('pole','框内为两根深色立杆及横梁，底部仅少量蓝色物体边缘，不是变压器。'),('building','框内主体为灰色建筑块的大平面与底座，上缘有细杆，不是目标设备。')],
2:[('building','灰色建筑块正侧面、深色矩形面板及底座完整可见。'),('building','同一灰色建筑块被另一seed判为开关柜；依据框内大块墙面和底座判断。')],
3:[('cabinet','蓝色普通柜体的平整侧面及黑色底座，非四类目标。')],
4:[('ground_shadow','框主要覆盖地面网格和长阴影，同时包含立杆下段及远处物体下缘；不是单个设备。')],
5:[('building','灰色建筑块侧面与右侧深色面板。'),('building','同一建筑块，右下角少量蓝色遮挡。'),('building','灰色建筑块的侧面、面板和底座，与O保留框重叠。')],
6:[('building','图像右缘截断的灰色建筑立面及深色矩形面板。'),('cabinet','蓝色普通柜体正面有灰色面板、底部有黑色基座。')],
7:[('cabinet','蓝色普通柜体正侧面及灰色前面板，框并非背景杆体。')],
8:[('mixed_structure','框内同时有灰色建筑立面、遮挡其前方的粗立杆和蓝柜一角，无法归为单一对象。')],
9:[('ground_shadow','宽框主要覆盖地面网格，左缘有立杆下段、顶部有围墙；不是开关柜。')],
10:[('building','灰色建筑大侧面为主体，右下少量蓝柜遮挡。'),('mixed_structure','宽框跨越灰色建筑、前方立杆及蓝柜上部，明显包含多个结构。')],
11:[('building','远处灰色建筑正面与深色矩形面板及基座，低冷光下仍可辨识。')],
12:[('building','灰色建筑正面及深色面板，被判电抗器。'),('building','同一建筑正面矩形面板被判电容器组，未见电容器阵列。'),('building','同一灰色建筑面板和下缘，另一seed也判电容器组。')],
13:[('building','灰色建筑正侧面、矩形暗面板与底座。'),('building','同一建筑块被判为开关柜，几何内容与前一框一致。')],
14:[('building','右侧被图像边界截断的灰色建筑面板及侧缘。'),('building','同一截断建筑，框另包含上方天空；主体仍为建筑边缘。'),('cabinet','蓝色普通柜体的侧面、部分前面板及基座。')],
15:[('building','灰色建筑背侧大平面及基座，右缘被立杆遮挡。')],
}

def main():
    path=OUT/'manifest.json';verify_tree(path);manifest=read(path);decisions=[]
    for f in manifest['frames']:
        observations=OBSERVATIONS[f['ordinal']]
        if len(observations)!=len(f['predictions']):raise ValueError('Missing explicit decisions')
        for kind in ('image','evidence'):
            if file_sha256(f[f'{kind}_path'])!=f[f'{kind}_sha256']:raise ValueError('Stale evidence')
        for p,(category,reason) in zip(f['predictions'],observations):
            decisions.append(dict(**p,frame_ordinal=f['ordinal'],view_id=f['view_id'],variant=f['variant'],
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],content_category=category,
                reason=reason,decision='reviewed',review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat()))
    if len(decisions)!=26 or len({r['prediction_id'] for r in decisions})!=26:raise ValueError('Review incomplete')
    transitions={str(s):dict(Counter(f'{int(r["O"])}->{int(r["N"])}' for r in manifest['frame_transitions'] if r['seed']==s)) for s in (7,17,27)}
    result=save(OUT/'visual-review.json',dict(status='reviewed',decisions=decisions,unique_images=15,prediction_observations=26,
        content_counts=dict(Counter(r['content_category'] for r in decisions)),
        same_class_O_overlap_counts=dict(Counter(str(r['corresponding_O_box']) for r in decisions)),
        frame_transitions=transitions,inputs={str(p):file_sha256(p) for p in [path,Path(__file__)]+[Path(f[k+'_path']) for f in manifest['frames'] for k in ('image','evidence')]}))
    print({k:result[k] for k in ('content_counts','same_class_O_overlap_counts','frame_transitions')})

if __name__=='__main__':main()
