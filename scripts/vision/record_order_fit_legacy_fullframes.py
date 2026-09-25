"""Authored full-frame decisions after individually viewing all 27 evidence pages."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior

NOTES={
 'C02':'前景大灰块与 control_building 对应；多数未标目标的投影落在这块建筑表面，没有看到独立目标内容。三处可见设备已有框；不把投影框认作透视可见证据。',
 'C03':'左后灰建筑可见，cabinet_center 和 capacitor_east 投影位于前景圆柱／变压器遮挡区；未见额外独立目标，三处设备已有框。',
 'C04':'中央较小蓝柜对应 cabinet_center，后灰块对应 control_building；其余可见设备及图缘片段已有框。',
 'C05':'画面只有独立封闭箱体、地面及围墙，箱体已有框；未见其他目标。',
 'C06':'变压器、后箱体和前面板柜均有框；其余是杆、围墙、地面及阴影。',
 'C09':'左后灰块为 control_building；两处未标目标投影分别落在圆柱和右侧柜体／变压器遮挡区，未见独立目标内容。五个可见设备均有框。',
 'C10':'下缘大蓝顶面与 cabinet_west 对应；五处可见目标及左缘截断变压器均有框。',
 'C12':'三处设备均有框，杆和围墙未见额外目标内容。',
 'C13':'下缘蓝色顶面与 cabinet_west 对应；其余四处设备均有框，右后柜受变压器遮挡但已有标签。',
 'C15':'后灰块对应 control_building，最右投影在图缘变压器表面，未见独立电容器内容；圆柱与图缘变压器均有框。',
 'C17':'三个设备均有框，杆穿过部分主体但未形成额外未框目标。',
 'C18':'switchgear_south 投影落在近变压器主体表面，没有独立可见柜体；三处可见目标已有框。',
 'C20':'灰色变体全图另有右下窄蓝片，位置和宽侧面对应 cabinet_center；五个灰色目标已有框，不把背景蓝片按颜色改称目标。',
 'C24':'全图为俯视圆柱及基座、围墙和地面，唯一目标已有框。',
 'C27':'中部蓝色小柜对应 cabinet_center，后灰块对应 control_building；三处可见目标已有框，保留近端大幅截断限制。',
 'C28':'中左面板小柜对应 cabinet_center，后灰建筑对应 control_building；两台变压器及后箱体均有框。',
 'C29':'右后蓝色部分对应 cabinet_center，后灰块对应 control_building；近变压器和远箱体均有框。',
 'C35':'左灰色带面板建筑对应 control_building，左下蓝色片段对应 cabinet_center；中部电容器已有框，未见额外目标。',
 'C36':'左缘大灰建筑对应 control_building，右前蓝块对应 cabinet_center；后变压器虽下部被普通柜体挡住仍已有框，另一箱体也有框。',
 'C39':'全图后排多个柜体、两台图缘变压器及右缘圆柱均有框；其余可见结构为杆、围墙和地面。保持逐标签截断限制。',
 'N01':'独立箱体顶面、正侧面和基座已框，周边只有地面和围墙。',
 'N03':'后变压器和前箱体均有框，未见其他独立目标；前后遮挡关系保留。',
 'N04':'左缘箱体、中央柜体、右变压器和后柜均有框，未见其他独立目标。',
 'N06':'背景变体逐图检查，右下蓝色窄片对应 cabinet_center；五处目标均有框，天空／地面变化不作为来源认证。',
 'N08':'常规颜色帧逐图检查，右下蓝色片段对应 cabinet_center；左侧柜群、变压器和圆柱均已有框。',
 'N09':'左后灰块为 control_building、左侧蓝面板柜为 cabinet_center；capacitor_east 投影位于变压器表面，未见独立内容；变压器已有框。',
 'R03':'四个柜体、右侧变压器及后方圆柱均有框，左下宽顶侧面属于已框近柜；未见其他独立目标。',
}


def run():
    ep=OUT/'legacy-fullframe-review-v1/evidence.json';dest=ep.parent/'review.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    e=prior.read(ep);prior.verify(e)
    if set(NOTES)!={m['review_id'] for m in e['members']}:raise ValueError('Missing or unexpected authored observation')
    paths=[ep,Path(__file__).resolve()];decisions=[]
    for m in e['members']:
        page=Path(m['page_path'])
        if prior.file_sha256(page)!=m['page_sha256']:raise ValueError('Stale viewed page')
        paths.append(page)
        decisions.append(dict(member_id=m['member_id'],review_id=m['review_id'],
            status='full_frame_supported_with_source_and_occlusion_limits',reason=NOTES[m['review_id']],
            full_frame_viewed=True,review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],page_sha256=m['page_sha256'],world_sha256=m['world_sha256'],
            pixel_visibility_certified=False,unseen_target_pixel_count='unknown',training_eligible=False))
    return prior.frozen(dest,dict(status='27_explicit_full_frame_decisions_recorded',decisions=decisions,
        limits='Visual full-frame inspection plus saved source context only; does not establish zero hidden pixels from AABB projections, nor revise existing per-label content limits.',
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['status'])
