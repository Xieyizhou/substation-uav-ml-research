"""Explicit visual observations; not a dataset relabeling or admission tool."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.build_retention_regression_review import OUT,read,save,file_sha256

OBS={
 'P01':'近景深色长方体主体、顶面三个浅色柱状附件及底座清楚可见；没有严重前景遮挡。',
 'P02':'后方灰绿色柜状主体部分被前景蓝绿色柜体遮住，下部受遮挡，上部宽侧面可见；不可据此认证唯一类别。',
 'P03':'灰色圆柱主体和完整方形底座明显可见，无明显前景遮挡；已有诊断为错类，不是目标完全不可见。',
 'P04':'远处灰色圆柱上部在柜体后露出，下部和底座被遮挡；图像尺度小，类别细节有限。',
 'P05':'光照变体中远处圆柱上部可见，下部被前景柜体遮挡；与原始条件分别检查，内容有限。',
 'P06':'中距离蓝绿色柜体有深色面板、宽侧面及底座可见；邻近大型设备占前景，但目标主体没有被严重遮挡。',
 'P07':'光照条件下目标蓝绿色柜体面板与侧面仍可见，颜色较暗；不是仅剩基座或边缘片段。',
 'P08':'远处蓝绿色长方体背侧、顶面及底座可见，尺度较小且看不到面板，类别判别线索有限。',
 'P09':'光照条件下远处柜状背侧及顶面可见，无显著前景遮挡；尺度和无面板视角可能增加难度。',
 'P10':'近景蓝绿色长方体背侧、顶面、底座清楚；面板不可见，但主体面积大，不是可见性缺失。',
 'P11':'灰色圆柱和方形基座清楚可见，无明显遮挡；此 seed 的诊断为低置信度，同图另一 seed 为错类。',
 'P12':'近景灰绿色宽侧面和黑色底座明显可见，表面有阴影；视觉观察不证明类别唯一性，诊断为错类。',
 'P13':'中近景深色长方体、三个浅色顶部附件和底座明显可见，无严重遮挡；诊断为同类低置信度。',
}

def validate(events,observations):
    ids=[e['event_id'] for e in events]
    if len(ids)!=len(set(ids)) or set(ids)!=set(observations):raise ValueError('Missing/duplicate review')
    for e in events:
        for kind in ('image','evidence'):
            if file_sha256(e[kind+'_path'])!=e[kind+'_sha256']:raise ValueError('Stale evidence')

def main():
    dest=OUT/'review.json'
    if dest.exists():raise ValueError('Preserve existing review')
    manifest=OUT/'manifest.json';events=read(manifest)['events'];validate(events,OBS)
    decisions=[{**e,'observation':OBS[e['event_id']],'review_nature':'AI辅助审核',
        'reviewed_at':datetime.now(timezone.utc).isoformat(),'status':'diagnostic_observation_not_label_certification'} for e in events]
    save(dest,dict(status='all_13_new_planned_miss_events_reviewed',decisions=decisions,
        unique_images=len({e['image_sha256'] for e in events}),
        limits='Does not exhaust non-planned full-image missed instances; no label changes or automatic training admission.',
        inputs={str(manifest):file_sha256(manifest),str(Path(__file__)):file_sha256(Path(__file__))}))
    print('REVIEWED',len(events),'unique images',len({e['image_sha256'] for e in events}))

if __name__=='__main__':main()
