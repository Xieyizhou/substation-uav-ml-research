"""Explicit observations from ten inspected triptychs, not automatic data approval."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.closed_gamma_design import OUT,prior

NOTES={
'G01':'俯视箱体顶面、三个浅色突出部和右侧箱体在三档均可见；侧面覆盖本来有限，未把顶视图认证为完整结构。',
'G02':'灰蓝顶面三个突出部及右侧箱体轮廓保留；明暗变化未新增遮挡，不能据此补认资产或像素可见性。',
'G03':'近处青色面板、顶部、基座及两个远处框内箱体仍可辨；暗档面板变暗但边界没有消失。',
'G04':'灰蓝面板外边界与细横纹仍可见，远处两块体保留；暗档纹理较弱，未认证全池监督质量。',
'G05':'封闭箱体顶部、正面及基座轮廓三档可辨；暗档正面仍与基座分离，不推断内部电容结构。',
'G06':'灰蓝箱体顶部、正面及底座保留，暗档面部更弱但外形边界可见；不声称新增结构证据。',
'G07':'圆柱椭圆顶面、侧壁、基座和阴影三档均可分辨；背景块体存在，本审核不重认证原标签覆盖。',
'G08':'近处圆柱顶部、侧壁、基座保留；暗档侧壁变暗但圆柱外轮廓仍可辨。',
'G09':'青色块体、左侧细杆、阴影与地面网格均保留；天空原已接近白色，不将预览视为新的空标签认证。',
'G10':'细杆与横件、小青色块体、地面阴影仍可见；上缘结构截断原已存在，不从外形推断资产身份。',
}

def validate(e,r):
    prior.verify(e);prior.verify(r)
    cards={c['id']:c for c in e['cards']}
    ds=r['decisions']
    if len(ds)!=len(cards) or len({d['id'] for d in ds})!=len(ds):raise ValueError('Missing/duplicate review')
    for d in ds:
        if d['id'] not in cards or d['evidence']!=cards[d['id']]:raise ValueError('Stale review evidence')
        if d['status']!='observed_no_new_augmentation_blocker' or not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Unresolved augmentation review')

def main():
    ep=OUT/'augmentation-review/evidence.json';dest=ep.with_name('review.json');e=prior.read(ep);prior.verify(e)
    if dest.exists():r=prior.read(dest);validate(e,r);return r
    if {c['id'] for c in e['cards']}!=set(NOTES):raise ValueError('Unexpected review inventory')
    now=datetime.now(timezone.utc).isoformat()
    r=prior.frozen(dest,dict(status='explicit_bounded_preview_review_complete_not_pool_admission',
        decisions=[dict(id=c['id'],evidence=c,reason=NOTES[c['id']],review_nature='AI辅助审核',review_time=now,status='observed_no_new_augmentation_blocker') for c in e['cards']],
        limits='Ten inspected representative triptychs only; not full-pool approval, source identity correction, instance-mask certification or proof of training benefit.',
        inputs={str(p):prior.file_sha256(p) for p in (ep,Path(__file__).resolve())}))
    validate(e,r);return r

if __name__=='__main__':print(main()['status'])
