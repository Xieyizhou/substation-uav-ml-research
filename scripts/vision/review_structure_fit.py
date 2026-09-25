"""New explicit observations from each displayed full frame and crop, not inherited passes."""
from datetime import datetime,timezone
from PIL import Image
from pathlib import Path
from scripts.vision.structure_fit import OUT,read,verify,frozen,file_sha256,validate_review,STRUCTURES

# Both lighting frames in each pair were separately displayed and inspected.
# Coordinates are normalized manual evidence ROIs, not revised training labels.
# Non-primary structures remain unknown unless separately documented below.
OBS={
1:('side_back base_fragment building_facade',[0,.20,.25,.96],'gray','none_apparent','left_edge','左缘灰块背侧及长基座；中间蓝块背面、三根杆与地面投影亦可见。'),
2:('side_back base_fragment building_facade',[.34,.39,.66,.74],'gray','none_apparent','none','中间灰块两侧大平面和底座完整；蓝块在右后，杆体与围墙可见。'),
3:('front_panel base_fragment building_facade',[.72,.12,1,1],'gray','none_apparent','right_bottom','右缘灰块正面深面板和基座被图缘截断；中央蓝柜背面，前方粗杆。'),
4:('side_back front_panel building_facade mixed_overlap',[.33,.34,.64,.72],'gray','foreground_pole','none','后方灰块宽侧、窄面板；粗杆挡面板一部分，前景蓝柜顶面与背面在右下出画。'),
5:('side_back top base_fragment',[.40,.39,.60,.65],'blue','none_apparent','none','蓝柜背侧两平面、顶面与基座清楚；左杆和投影、墙地边界。'),
6:('front_panel side_back top base_fragment',[.43,.40,.59,.63],'blue','none_apparent','none','蓝柜正面深面板、窄侧面、顶面和基座完整；右侧细杆。'),
7:('thin_pole_crossbar',[.46,.24,.54,.75],'dark','none_apparent','none','单根中景杆及短横件完整，围墙、地面网格、杆影，无柜状块体。'),
8:('thin_pole_crossbar',[.46,.26,.54,.75],'dark','none_apparent','none','单根细杆与横件，墙角与地面杆影，无柜状块体。'),
9:('thin_pole_crossbar',[.51,.25,.56,.75],'dark','none_apparent','none','中间细竖杆，上方导线横梁及悬挂小件；地面投影，未见柜体或建筑立面。'),
10:('side_back top base_fragment',[.44,.41,.57,.62],'blue','none_apparent','none','远处蓝柜背侧、顶面和基座；左缘粗杆上下截断，周围横梁及杆影。'),
11:('side_back top base_fragment',[.43,.40,.57,.62],'blue','none_apparent','none','蓝柜两无面板侧面、顶面与基座，后方框架杆与导线。'),
12:('thin_pole_crossbar',[.45,.24,.51,.77],'dark','none_apparent','none','中间细竖杆和邻近框架，顶部导线及悬件，地面投影；没有柜状块体。'),
13:('thin_pole_crossbar',[.47,.28,.52,.72],'dark','none_apparent','none','中央细杆完整，左缘灰建筑面板和蓝柜截片；杆影落于网格。'),
14:('side_back base_fragment building_facade',[.16,0,.79,.62],'gray','none_apparent','top','俯视灰块下侧及黑色基座转角，上部出画；周围地面网格。'),
15:('shadow wall_sky_ground',[.19,.34,1,1],'gray_dark','none_apparent','none','地面长杆影、网格和右侧墙影；上缘含建筑及蓝柜底部。'),
16:('side_back top base_fragment building_facade',[.27,.25,.73,1],'gray','none_apparent','none','大灰块两侧立面、宽顶面和基座清楚；旁边杆体在图缘。'),
17:('front_panel base_fragment building_facade',[.52,.10,1,1],'gray','none_apparent','right_bottom','右缘近景灰建筑深面板与底座出画，左侧蓝柜背面清楚。'),
18:('side_back top base_fragment building_facade',[.36,.35,.64,.74],'gray','none_apparent','none','中距灰块两背侧、顶部和底座，前方粗杆未遮住该主体。'),
19:('front_panel side_back top base_fragment',[.40,.39,.59,.65],'blue','none_apparent','none','蓝柜正面深面板、宽侧面、顶面和基座清楚；后方灰建筑被杆体遮挡。'),
20:('side_back base_fragment building_facade',[.71,.35,1,.76],'gray','none_apparent','right','右缘灰建筑宽背侧和基座截断，左侧天空与围墙。'),
21:('side_back building_facade mixed_overlap',[.36,.36,.65,.69],'gray_dark','foreground_pole','none','灰建筑背侧被近景粗杆遮住中央窄条，右侧蓝柜背面；不能混作一个对象。'),
22:('wall_sky_ground',[0,0,1,1],'gray','none_apparent','none','全图为地面网格与宽浅色地面标线；未见竖直柜状主体。'),
23:('side_back top base_fragment building_facade',[0,.34,.35,.97],'gray','none_apparent','left','左缘灰建筑背侧、顶面、基座，远处细杆和墙边。'),
24:('front_panel top base_fragment building_facade',[.74,.37,1,.80],'gray','none_apparent','right','右缘灰建筑深面板和顶面底座；中央前景蓝柜面板亦可见。'),
}

OBS.update({
25:('front_panel side_back top base_fragment mixed_overlap',[.28,.37,.64,1],'blue_gray','foreground_pole_and_cabinet','bottom','前景蓝柜正侧面、顶面与底座下缘出画；后方灰面板被杆和柜体部分遮挡。'),
26:('thin_pole_crossbar',[.47,.27,.52,.73],'dark','none_apparent','none','中间细杆及顶部短件，左侧灰建筑背面截断、地面短投影。'),
27:('front_panel top base_fragment building_facade',[.31,.25,.75,.92],'gray','none_apparent','none','灰建筑正面面板、宽顶面和黑基座完整；左侧投影。'),
28:('shadow wall_sky_ground',[.06,.49,1,1],'gray_dark','none_apparent','none','地面网格、浅色引导线和杆影通向后景；左侧墙边界，背景有灰建筑和蓝柜。'),
29:('side_back top base_fragment building_facade',[.33,.12,.67,.60],'gray','none_apparent','none','中距灰块两侧、顶面及底座转角清楚，地面和墙影围绕。'),
30:('front_panel side_back base_fragment building_facade',[.36,.36,.62,.68],'gray','none_apparent','none','灰建筑正面深色面板、右窄侧与底座完整；左粗杆未遮住主体。'),
31:('side_back front_panel base_fragment building_facade',[.70,.35,1,.77],'gray','none_apparent','right','右缘灰建筑宽侧和窄面板、基座；最右下蓝柜局部另计未知关系。'),
32:('side_back base_fragment building_facade',[.58,.24,1,1],'gray','none_apparent','right_bottom','右缘近景灰建筑大平面与底座被画面截断，左侧围墙天空地面。'),
33:('side_back top base_fragment building_facade',[.49,.17,1,1],'gray','none_apparent','right_bottom','近景灰建筑宽背面、顶面和底座转角；右下出画。'),
34:('shadow wall_sky_ground',[0,0,1,1],'gray_dark','none_apparent','none','主要地面网格、白线和右上墙影；不把网格平面记作设备面板。'),
35:('side_back base_fragment building_facade',[.15,.17,.83,1],'gray','none_apparent','bottom','大尺度灰建筑单一背面，底座贴下边；后方立杆下部被遮。'),
36:('side_back top base_fragment building_facade',[.73,.38,1,.80],'gray','none_apparent','right','右缘灰建筑背面、窄顶面、基座；中央近景杆上下出画但未遮住主ROI。'),
})
OBS.update({
37:('front_panel side_back top mixed_overlap',[.20,.37,.64,1],'blue_gray','foreground_cabinet_and_pole','bottom','近景蓝柜底部出画，后景灰建筑面板与前景杆部分重叠；两帧均查看。'),
38:('side_back top base_fragment building_facade',[.38,.37,.61,.70],'gray','none_apparent','none','灰建筑背面和顶面完整，旁边蓝柜及杆体多尺度同框，但未见主ROI明显被遮。'),
39:('side_back base_fragment building_facade',[.32,.31,.68,.78],'gray','none_apparent','none','灰建筑两个宽背侧和基座完整；后方杆下部被建筑遮挡。'),
40:('front_panel side_back top base_fragment building_facade',[.37,.34,.61,.71],'gray','none_apparent','none','右侧中景灰面板、窄侧、顶部和底座；左前粗杆不当作建筑。'),
41:('side_back base_fragment building_facade shadow',[.28,.25,.81,1],'gray_dark','none_apparent','none','灰建筑大背面与基座完整，下方矩形阴影明显。'),
42:('side_back front_panel top base_fragment building_facade mixed_overlap',[.35,.35,.62,.71],'gray','foreground_pole','none','灰建筑宽侧及右面板，细杆遮面板右段；前景蓝柜下缘出画。'),
43:('side_back base_fragment building_facade',[.34,.32,.70,.76],'gray','none_apparent','none','中景灰块宽背侧与窄侧、基座；后方杆下部被建筑遮住。'),
44:('side_back top base_fragment building_facade',[.25,.24,.68,.96],'gray','none_apparent','none','较高视角灰建筑宽顶面、背面、基座；右后蓝柜。'),
45:('side_back front_panel base_fragment building_facade',[.29,.25,.76,.95],'gray','none_apparent','none','灰建筑宽侧面和左边窄正面深面板，基座清楚。'),
46:('thin_pole_crossbar',[.47,.28,.54,.72],'dark','none_apparent','none','中央细杆与短横件完整，左远处长蓝柜侧面及面板；左缘另一杆。'),
47:('shadow',[.20,.41,.53,.54],'dark','none_apparent','none','中央杆底向左形成细长阴影，周围地面网格；左缘蓝柜截片。'),
48:('shadow',[.38,.08,.52,.38],'dark','none_apparent','none','俯视杆底和纵向长阴影，网格及浅色标线；杆上部出画。'),
})
OBS.update({
49:('side_back top base_fragment',[.69,.36,1,.84],'blue','none_apparent','right','右缘长蓝柜背侧、顶面和底座截断，中央杆和墙边。'),
50:('side_back top base_fragment',[.35,.36,.64,.68],'blue','none_apparent','none','蓝柜宽背面、薄顶面和底座完整；背景围墙。'),
51:('side_back top base_fragment',[.74,.39,1,.88],'blue','none_apparent','right','右缘近景蓝柜宽背面与顶面、底座，右侧出画。'),
52:('shadow wall_sky_ground',[.03,.45,1,1],'gray_dark','none_apparent','none','地面网格和白线、杆影；远处蓝柜中央被杆挡住。'),
53:('side_back top base_fragment mixed_overlap',[.61,.34,1,.96],'blue','adjacent_cabinet','none','右侧两个蓝柜相互重叠，可见背侧、顶面和各自底座；地面长线影。'),
54:('thin_pole_crossbar',[.47,.21,.52,.79],'dark','none_apparent','none','中间细杆、上方线与悬件、右框架杆；没有柜状主体。'),
55:('top shadow',[.34,.65,.65,1],'blue_dark','none_apparent','bottom','近景蓝柜顶面出画，细线影横过顶面；右后另一蓝柜面板完整。'),
56:('shadow',[.20,0,.49,.10],'dark','none_apparent','top','上缘杆底及水平影，地面网格和直角白线占大部。'),
57:('side_back top base_fragment shadow',[.40,.35,.61,.72],'blue_dark','none_apparent','none','蓝柜宽侧、顶面和基座，长细影经过左缘及地面；另一柜在远处。'),
58:('front_panel side_back top base_fragment',[.74,.40,1,.80],'blue','none_apparent','right','右缘蓝柜面板、宽侧面、顶面和基座，框外为天空墙及白线地面。'),
59:('front_panel side_back top base_fragment',[.59,.42,.70,.57],'blue','none_apparent','none','远处右蓝柜面板、侧面及基座可辨；左蓝柜被前景杆挡一条。'),
60:('shadow wall_sky_ground',[.34,0,.62,1],'gray_dark','none_apparent','none','左侧墙沿、地面及长墙影；右上有杆底与橙色标记，不是设备。'),
})
ABSENT_PAIRS={7,8,9,12,22,34,54,56,60}
REACT={
'T01':('partial_identifiable','foreground_pole','none','远处圆柱宽弧面及部分基座可辨，前景杆遮右部。'),
'T02':('clear_body','none_apparent','none','完整圆柱宽主体与方形基座清楚。'),
'T03':('clear_body','none_apparent','none','远处圆柱顶面、主体和基座完整。'),
'T04':('partial_identifiable','minor_foreground_equipment','right','右缘圆柱宽弧面、部分顶面与基座可辨，但图缘截断。'),
'T05':('partial_identifiable','foreground_equipment','none','远处圆柱主体部分可辨，下右部被变压器遮挡。'),
'T06':('clear_body','none_apparent','none','近景圆柱完整宽主体与基座。'),
'T07':('clear_body','none_apparent','none','大圆柱主体和方形基座完整，顶部细节有限但主体明确。'),
'T08':('insufficient_or_uncertain','foreground_equipment','left','左图缘只见受遮挡圆柱上部窄片，白色附件与前景设备混入；内容不足，不认证可见性。'),
'T09':('clear_body','none_apparent','none','中远距圆柱椭圆顶面、主体和基座清楚。'),
'T10':('clear_body','none_apparent','none','近景圆柱弧面和基座完整。'),
'T11':('partial_identifiable','foreground_equipment','none','后方灰圆柱主体右部可辨，左侧被灰设备遮住。'),
'T12':('partial_identifiable','foreground_cabinet','none','圆柱顶面和大部上身可辨，下方被蓝柜遮挡。'),
'T13':('partial_identifiable','foreground_transformer','none','中性灰圆柱顶面、上身和部分基座可辨，下右被设备遮挡。'),
'T14':('partial_identifiable','foreground_cabinet','none','中性灰变体中圆柱上身可辨，下部被柜体挡住。'),
'T15':('partial_identifiable','foreground_transformer','none','背景变体中圆柱上部及部分基座可辨，下右被变压器遮挡。'),
'T16':('partial_identifiable','foreground_cabinet','none','背景变体中圆柱上部可辨，下部被蓝柜遮挡。'),
'T17':('partial_identifiable','foreground_equipment','none','后方圆柱宽弧面右部和基座可辨，左部被前景蓝设备遮挡。'),
'T18':('partial_identifiable','foreground_equipment','none','背景变体，左部仍被蓝设备遮挡，圆柱主体右部可辨。'),
'T19':('partial_identifiable','foreground_transformer','none','原始条件圆柱顶面及上身、部分基座可辨，下右被变压器遮挡。'),
'T20':('clear_body','none_apparent','none','正面圆柱完整宽身和基座，无遮挡。'),
'T21':('partial_identifiable','foreground_cabinet','none','圆柱上部宽弧面可辨，下半及基座被近景柜顶遮住。'),
'T22':('clear_body','none_apparent','none','俯视完整顶面、圆柱主体与方形基座。'),
'T23':('clear_body','none_apparent','none','中距完整圆柱和基座，右侧杆不挡主体。'),
'T24':('clear_body','none_apparent','none','远处完整圆柱上身至基座均可见。'),
'T25':('clear_body','none_apparent','none','中距圆柱顶面、主体与基座完整。'),
'T26':('partial_identifiable','minor_foreground_building','right','右缘圆柱大部弧面可辨，右下被建筑角遮住并出画。'),
'T27':('clear_body','none_apparent','none','中距圆柱主体和基座完整。'),
'T28':('clear_body','none_apparent','none','近景圆柱顶面、弧面和基座完整。'),
'T29':('insufficient_or_uncertain','foreground_pole_and_equipment','none','只见杆和变压器之间窄灰色圆柱条及少量底座，类别内容不足。'),
'T30':('insufficient_or_uncertain','foreground_equipment','left_extreme','极窄左图缘标签，放大仍主要是前景蓝设备边缘；不能确认可辨识电抗器内容。'),
'T31':('clear_body','none_apparent','none','近景完整圆柱宽弧面、顶面与基座。'),
'T32':('partial_identifiable','foreground_transformer','right','右图缘上半圆柱宽弧面和顶面可辨，下部被前景变压器挡住。'),
'T33':('insufficient_or_uncertain','foreground_transformer','none','仅后方圆柱上缘窄带，前景变压器主体和浅色附件占框大部；内容不足。'),
'T34':('clear_body','none_apparent','none','远处圆柱主体、顶面及基座清楚可辨。'),
'T35':('clear_body','none_apparent','none','近景椭圆顶面、完整圆柱和方形基座清楚。'),
'D01':('clear_body','none_apparent','none','原始条件圆柱主体及基座完整，无明显遮挡。'),
'D02':('clear_body','none_apparent','none','光照条件圆柱变暗但主体和基座完整。'),
'D03':('insufficient_or_uncertain','foreground_cabinet','none','原始条件只露出远处圆柱上部，下部及基座被柜体遮挡；细节不足。'),
'D04':('insufficient_or_uncertain','foreground_cabinet','none','光照条件上部短片更暗，前景柜体挡住下部；保留内容不足状态。'),
'D05':('partial_identifiable','none_apparent','right','原始条件右缘圆柱宽弧面、顶面及部分基座可辨，非仅窄边。'),
'D06':('partial_identifiable','none_apparent','right','光照条件右缘圆柱弧面及基座可辨，右侧仍被图缘截断。'),
'D07':('clear_body','none_apparent','none','原始条件近景大圆柱完整，主体、顶面与基座清楚。'),
'D08':('clear_body','none_apparent','none','光照条件近景圆柱完整，虽变暗仍非不可见。'),
'D09':('partial_identifiable','foreground_transformer','none','原始条件后方圆柱顶面及上身可辨，下半被变压器遮住，浅色附件在框内。'),
'D10':('partial_identifiable','foreground_transformer','none','光照条件圆柱顶面及上身可辨，下方被设备挡住；附件不认证为目标组件。'),
}

def main():
    ep=OUT/'evidence.json';p=read(OUT/'protocol.json');e=read(ep);verify(e)
    if len(OBS)!=60 or len(REACT)!=45:raise ValueError('Explicit observations incomplete')
    events={x['event_id']:x for x in e['events']};ds=[]
    for n in p['negative']:
        eid=n['event_id'];idx=(int(eid[1:])+1)//2;features,box,color,oc,tr,reason=OBS[idx];ev=events[eid]
        with Image.open(ev['image_path']) as im:w,h=im.size
        roi=[round(v*(w if i%2==0 else h)) for i,v in enumerate(box)]
        states={k:dict(state='unknown',reason='全图已查看；该结构未单独标定，不能用主ROI推断不存在。') for k in STRUCTURES}
        for k in features.split():states[k]=dict(state='present',roi_xyxy=roi,reason=reason,color_appearance=color,occlusion=oc,truncation=tr,asset_identity='unknown_pending_world_crosscheck')
        if idx in ABSENT_PAIRS:
            for k in ('front_panel','side_back','top','base_fragment','building_facade'):states[k]=dict(state='not_seen',reason='逐图全景中只见杆架、墙和地面，未见该块体结构。')
        ds.append(dict(**{k:ev[k] for k in ('event_id','source_event_identity','image_sha256','evidence_sha256')},
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),full_image_inspected=True,
            reason=n['source']['lighting_id']+' 本帧全图及局部已单独查看：'+reason,structures=states,
            status='reviewed_with_named_fine_structure_gaps',pixel_visibility_certified=False))
    for eid,(content,oc,tr,reason) in REACT.items():
        ev=events[eid];ds.append(dict(**{k:ev[k] for k in ('event_id','source_event_identity','image_sha256','evidence_sha256')},
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),full_image_inspected=True,
            reason=reason,identifiable_content=content,occlusion=oc,truncation=tr,component_identity='unknown',pixel_visibility_certified=False,status='reviewed'))
    validate_review(e,ds)
    frozen(OUT/'review.json',dict(status='reviewed_explicit_unknowns_retained',decisions=ds,inputs={str(ep):file_sha256(ep),str(Path(__file__)):file_sha256(Path(__file__))}))

if __name__=='__main__':main()
