"""Explicit observations after individually inspecting pilot evidence 000–023."""
from datetime import datetime, timezone
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.hard_negative_coverage import OUT,read,save,verify,file_sha256

# ROI coordinates refer to displayed 1280x720 image, without the 36px header.
# Both light images in each entry were separately inspected, not auto-approved.
OBS = {
0: ('building',[475,268,810,514],'建筑背面、侧面和底座完整，右侧多根杆体及柜体可辨；未见四类目标'),
1: ('building',[469,273,776,487],'建筑前面板、侧面和底座可辨；近景杆体邻接左侧，主体完整'),
2: ('building',[667,80,1280,720],'右缘截断的大尺度建筑前面板与底座；左侧为普通蓝色柜体和杆体'),
3: ('building',[919,262,1280,534],'右缘截断的建筑背面与底座，地面边界及着陆标记可见'),
4: ('ground_shadow',[0,72,1280,720],'俯视网格覆盖全图主体，杆体底部及长投影在上中部；无目标'),
5: ('ground_shadow',[0,290,838,720],'地面网格与多根杆影横穿，右侧普通柜体底座交界清晰'),
6: ('mixed_structure',[366,35,1280,720],'建筑遮挡后方杆体下部，右侧近景杆体及柜体边缘共同出现；混合组成可辨'),
7: ('mixed_structure',[0,0,778,720],'近景杆体遮挡/邻接后方建筑，左侧普通柜体与地面组成多尺度结构'),
8: ('cabinet',[452,270,819,482],'完整普通柜体背面、顶部与底座，身份为非目标 cabinet_west'),
9: ('cabinet',[441,468,809,720],'前景普通柜体在下缘截断，旁有杆体及投影，远处另一普通柜体完整'),
10: ('pole',[604,215,679,500],'中景完整杆体及横杆，左侧近景杆体在边缘截断，普通柜体在后方'),
11: ('pole',[626,207,654,508],'正对窄杆体，横杆沿视线投影近乎重合，底部与投影清楚；左缘建筑/柜体截断'),
}

def main():
    p=OUT/'pilot-review-manifest.json';m=read(p);verify(m)
    if len(m['frames'])!=24:raise ValueError('Unexpected pilot')
    decisions=[]
    for r in m['frames']:
        category,box,reason=OBS[r['review_index']//2]
        decisions.append(dict(view_id=r['view_id'],image_sha256=r['image_sha256'],decision='accepted',
            no_target_visible=True,coverage_confirmed=True,review_nature='AI-assisted',reviewer='Codex individual visual inspection',
            reviewed_at=datetime.now(timezone.utc).isoformat(),reason=f'{r["variant"]} 已单独查看全图：{reason}',
            rois=[dict(content=category,bbox_xyxy=[round(v*1.5) for v in box],visibility='visible',
                       truncation='image_edge' if r['coverage_unit'] in ('B2','C2') else 'see_reason')]))
    save(OUT/'pilot-decisions.json',dict(decisions=decisions,status='explicitly_reviewed',inputs={str(p):file_sha256(p),str(Path(__file__)):file_sha256(Path(__file__))}))

if __name__=='__main__':main()
