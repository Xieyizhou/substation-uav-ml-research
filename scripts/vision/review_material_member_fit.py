"""Explicit current-task visual observations, not training approvals."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.material_member_fit import OUT,preflight,prior
from src.ml.artifacts import object_sha256
NOTES='''
M01-00|partial|蓝黑变压器左缘截断；宽两侧面、三端子及基座可辨，暗天空不抹掉主体边界
M01-01|clear|远处电容器壳体完整矩形及基座可辨，斜阴影不遮外轮廓；不把后方建筑面板当作设备内部
M02-00|partial|三端子和宽主体在，右侧被黑杆遮挡一角
M02-01|partial|近变压器左下出画，仍有宽顶面、侧面和端子，不是仅基座
M02-02|partial|电容器下左被前景蓝柜挡住，大部宽侧面和顶面可辨
M03-00|partial|后方灰柜剩宽背侧面，上边和右界与邻柜接近；前柜遮住下部，未见面板正面
M03-01|partial|宽灰色背面被前杆穿过，底座可见；背向端子，无独立类别外观认证
M03-02|partial|目标是前景蓝柜后、变压器左侧的灰色背面；宽矩形残部可辨但灰色相接，不能认证像素归属
M03-03|partial|灰圆柱右侧宽曲面和基座可辨，左下被近变压器挡住
M04-00|partial|后柜下半被近柜挡，灰顶盖、部分面板及侧面可辨，面板线弱
M04-01|partial|左边出画，灰柜背侧宽面与基座在，没有面板正面证据
M04-02|clear|完整灰变压器主体、基座和三端子清楚，前杆未跨主体
M04-03|clear|完整灰柜主体、细面板边线、顶面与基座可辨；面板对比弱但非消失
M04-04|partial|圆柱大部及顶椭圆可辨，右下被前变压器遮挡
M05-00|partial|暗背景中蓝黑变压器宽面和三端子仍在，右侧被杆挡
M05-01|partial|前景变压器左下出画，蓝灰顶侧面和端子保留
M05-02|partial|蓝灰电容器壳体左下被蓝柜挡，宽主体和上边可分
M06-00|partial|灰变压器宽主体、顶端子可辨，右杆遮部分侧面
M06-01|partial|灰近变压器左下截断，宽顶面与端子可辨
M06-02|partial|灰电容器壳体大部可辨，左下前景蓝柜遮挡；后建筑不是目标面板
M07-00|partial|后柜蓝顶和深色面板上部可辨，下部被近柜挡住
M07-01|partial|左缘截断蓝柜宽背侧面与基座，褐地不影响边界
M07-02|clear|完整蓝黑变压器及三端子、基座清楚
M07-03|clear|蓝柜完整主体，深面板与蓝框分离清楚
M07-04|partial|灰圆柱宽曲面与顶椭圆在，右下被蓝变压器遮住
M08-00|partial|后蓝柜宽背面可辨，下部被前柜挡，邻柜上缘相接
M08-01|partial|蓝灰变压器宽背面被黑杆穿过，底座在，未朝向端子
M08-02|partial|目标蓝背面位于前景蓝柜后，右侧被近变压器挡；矩形残部可辨，不认证掩码
M08-03|partial|灰圆柱宽右侧面与底座在，左下有前景变压器遮挡
M09-00|partial|暗天空前仍见后蓝柜宽背面，前景遮下部，邻柜相接
M09-01|partial|蓝灰背面和基座可辨，但黑杆遮中部
M09-02|partial|前蓝柜上方的目标背面残部可辨，右侧变压器遮挡；不能由裁剪内全部像素认定目标
M09-03|partial|圆柱右曲面和底座可辨，左侧被前景遮挡，天空对比稍弱
M10-00|partial|灰变压器左缘截断，宽两面、三端子和基座可辨
M10-01|clear|完整灰电容器壳体、基座清楚，斜阴影存在但不遮轮廓
M11-00|partial|蓝黑变压器左缘截断，宽主体及三端子可辨
M11-01|clear|完整蓝灰壳体及基座；后方建筑不计作设备面板，未见前景遮挡主体
M12-00|partial|后蓝柜下半被近柜挡，顶盖和部分深面板可辨
M12-01|partial|左缘截断，宽蓝柜背侧与基座在
M12-02|clear|完整蓝黑变压器及三端子、基座可辨
M12-03|clear|完整蓝柜与深面板、顶面、基座均可见
M12-04|partial|圆柱大部与顶面在，右下被近变压器遮住
'''

def validate(p,r):
    prior.verify(r)
    if r['protocol_identity']!=p['identity']:raise ValueError('Stale review protocol')
    ds={d['event_id']:d for d in r['decisions']}
    if len(ds)!=len(r['decisions']) or set(ds)!={e['event_id'] for e in p['events']}:raise ValueError('Missing/duplicate review')
    for e in p['events']:
        if ds[e['event_id']]['evidence_sha256']!=object_sha256(e):raise ValueError('Stale review evidence')
    return ds

def main():
    p=preflight();dest=OUT/'review.json'
    if dest.exists():validate(p,prior.read(dest));return
    events={x['event_id']:x for x in p['events']};ds=[]
    for line in NOTES.strip().splitlines():
        eid,status,reason=line.split('|');ds.append(dict(event_id=eid,content=status,reason=reason,evidence_sha256=object_sha256(events[eid]),
            reviewed_at=datetime.now(timezone.utc).isoformat(),review_nature='AI辅助审核',pixel_visibility_certified=False,training_approved=False))
    r=prior.frozen(dest,dict(status='explicit_content_review_complete_diagnostic_only',protocol_identity=p['identity'],decisions=ds,
        full_frame_note='12 full images inspected alongside own crops. No new definite unboxed target claimed; this is not mask-certified exhaustive absence or a training approval.',
        inputs={str(x):prior.file_sha256(x) for x in (OUT/'protocol.json',Path(__file__).resolve())}))
    validate(p,r);print('REVIEWED',len(ds))

if __name__=='__main__':main()
