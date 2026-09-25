"""Explicit per-image review transcript for remaining images 000–071."""
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.hard_negative_coverage import OUT,read,save,verify,file_sha256

# Pair indexes map two separately inspected lighting images to common geometry.
OBS={
0:('building',[425,243,866,550],'建筑背面及侧面完整，后方杆体下部被建筑遮挡'),
1:('building',[472,269,797,505],'建筑侧面与前面板可辨，右前杆体遮挡一窄条，建筑不出画'),
2:('building',[389,191,965,702],'大尺度建筑侧面及前面板边缘、底座完整'),
3:('building',[387,195,960,642],'正对建筑背面与底座，左方地面着陆标记'),
4:('building',[420,194,932,651],'俯视建筑前面板、顶部和底座完整'),
5:('building',[342,193,855,677],'较高视点建筑背面、顶部、底座完整，右后柜体杆体可辨'),
6:('building',[909,265,1280,543],'右缘建筑截断，面板局部和侧面可见，前景柜体也在右缘'),
7:('building',[0,251,446,680],'左缘建筑背面/侧面截断，远处杆体与地面'),
8:('building',[951,278,1280,565],'右缘建筑面板截断，普通柜体和杆体在前方'),
9:('building',[632,145,1280,720],'近距离右缘和下缘截断的建筑背面及顶部'),
10:('building',[751,179,1280,720],'右缘大尺度建筑背面和底座，左侧着陆标记及网格'),
11:('building',[953,280,1280,567],'右缘建筑背面截断，中央近景杆体上下出画'),
12:('ground_shadow',[151,0,1280,720],'斜俯视网格和转折白线，顶缘杆体底部与投影'),
13:('ground_shadow',[0,409,1280,720],'建筑底座近景在上半部，底缘大面积网格与斜向地面线'),
14:('ground_shadow',[0,0,1280,720],'全图地面网格和斜向白色地面标线，无目标'),
15:('ground_shadow',[531,317,1280,720],'边界墙阴影、网格及橙色着陆标记，非目标设施'),
16:('ground_shadow',[0,0,991,720],'地面网格和白线为主体，右上边界墙及阴影'),
17:('ground_shadow',[154,282,1280,720],'杆体底部及横向投影、网格，左缘普通柜体'),
18:('ground_shadow',[0,275,945,720],'网格与中间杆影清楚，右侧墙影、上缘建筑及柜体底座'),
19:('ground_shadow',[247,321,1280,720],'远景杆体投影与网格白线，建筑和普通柜体在后方'),
20:('ground_shadow',[422,276,920,471],'建筑底座与地面交界及阴影；周围杆影和大面积网格'),
21:('ground_shadow',[0,292,1009,720],'杆影、白线和网格，远处普通柜体与立杆'),
22:('mixed_structure',[365,0,813,720],'近景柜体遮挡杆体底部和建筑下部，三种结构实际重叠'),
23:('mixed_structure',[194,0,1058,720],'建筑近景遮挡后方柜体大部与杆体下部，左缘柜体窄条及顶部杆体可辨'),
24:('mixed_structure',[489,98,1280,720],'建筑后景、杆体与普通柜体前景多尺度同框；无明显互相遮挡，不标为遮挡子型'),
25:('mixed_structure',[306,130,872,545],'建筑遮挡后方两根杆体下部，左侧另一杆体及地面组成结构；柜体不可见'),
26:('building',[387,194,960,642],'仅建筑前面与底座清楚，其他预估结构完全被遮挡，不满足混合结构覆盖'),
27:('mixed_structure',[488,0,1280,720],'近景杆体遮挡建筑中央，右侧柜体与另两根杆体可辨'),
28:('cabinet',[534,259,760,515],'前景普通柜体完整，另一个柜体在后方，杆影穿过柜体及地面'),
29:('cabinet',[533,287,749,461],'完整普通柜体前面板及底座，后方杆体和建筑'),
30:('cabinet',[765,312,870,388],'远处完整普通柜体；中间另一柜体被杆体部分遮挡，上部为既有支架导线结构'),
31:('cabinet',[16,323,265,458],'左側普通柜体实际仍完整，未接触边缘且无明显遮挡，不满足C2'),
32:('cabinet',[896,267,1280,598],'右缘截断的长普通柜体背面、侧面和底座'),
33:('cabinet',[962,299,1280,544],'右缘截断普通柜体前面板与侧面，后方为既有支架'),
34:('pole',[624,165,650,561],'中景窄杆体、右缘近景支架及上方导线组件，地面投影；非四类目标'),
35:('pole',[621,207,655,514],'中景窄杆体完整，左缘建筑截断并遮挡后方立杆下部'),
}

def main():
    p=OUT/'remaining-review-manifest.json';m=read(p);verify(m)
    if len(m['frames'])!=72:raise ValueError('Unexpected remaining set')
    decisions=[]
    for r in m['frames']:
        kind,box,reason=OBS[r['review_index']//2];held=r['ordinal'] in (34,41)
        decisions.append(dict(view_id=r['view_id'],image_sha256=r['image_sha256'],decision='held' if held else 'accepted',
            no_target_visible=True,coverage_confirmed=not held,review_nature='AI-assisted',reviewer='Codex individual visual inspection',
            reviewed_at=datetime.now(timezone.utc).isoformat(),reason=f'{r["variant"]} 已单独查看：{reason}',
            rois=[dict(content=kind,bbox_xyxy=[round(v*1.5) for v in box],visibility='visible',truncation='see_reason')]))
    save(OUT/'remaining-decisions.json',dict(status='reviewed_with_coverage_holds',accepted=68,held=4,decisions=decisions,
        inputs={str(p):file_sha256(p),str(Path(__file__)):file_sha256(Path(__file__))}))

if __name__=='__main__':main()
