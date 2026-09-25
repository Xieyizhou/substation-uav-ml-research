"""Explicit posttraining observations; never emits model acceptance."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from scripts.vision.audit_order_training import OUT, SOURCE, PRIOR, ROOT, read, file_sha256, frozen, verify_tree
from scripts.vision.exposure_order_retention import baseline_verify
from scripts.vision.review_exposure_order_retention import validate_decisions
from src.ml.artifacts import object_sha256

# Independently viewed full-image and crop panels F01–F42 this review.
OBS = [
('混合结构','天空占大部，下方围墙、右侧粗杆。'),
('混合结构','天空、底部围墙地面、左缘窄杆。'),
('混合结构','左上天空和围墙，右缘贴杆，与 F01 重叠。'),
('混合结构','天空、围墙和右杆，不是中间蓝柜。'),
('混合结构','暗光天空、围墙及右侧宽杆片段。'),
('混合结构','与 F05 同一区域，天空围墙和粗杆。'),
('混合结构','斜向围墙、天空及少量网格地面。'),
('柜体','左图缘截断蓝绿色柜体背侧、顶面与底座。'),
('混合结构','天空、左杆上部、围墙和右下灰柜片段。'),
('柜体','蓝柜正面深色面板、右侧和基座。'),
('柜体','较远蓝柜正面面板、侧面和基座。'),
('杆体','两根竖杆及横件主导，底部有墙与柜缘。'),
('柜体','蓝柜宽侧面、窄面板、顶面及基座。'),
('混合结构','天空、右斜杆上部及底部围墙。'),
('柜体','灰柜背侧下半部及黑基座，上半部不在框内。'),
('柜体','灰柜深色正面面板、宽侧面、顶面和基座。'),
('柜体','远处灰柜宽侧与右侧深色面板，杆仅在边缘。'),
('柜体','暗光灰柜侧面、面板和基座仍明确。'),
('柜体','中距灰柜正面深色矩形面板及基座。'),
('柜体','暗光同一灰柜正面面板和基座。'),
('柜体','暗光左侧蓝柜正面面板、侧面和底座。'),
('柜体','灰柜宽背侧与右面板，黑色基座。'),
('柜体','灰柜无面板宽面及基座，右部被粗前景杆遮挡。'),
('柜体','蓝柜正面面板、右侧与基座，与 F10 同图结构。'),
('柜体','远处蓝柜面板、侧面及基座，与 F11 同图结构。'),
('柜体','灰柜宽侧和右面板、基座；非邻近杆。'),
('柜体','暗光灰柜宽侧、右面板和基座。'),
('混合结构','斜围墙占主部，上方天空、下方网格及阴影。'),
('柜体','灰柜正面深面板和黑基座。'),
('柜体','暗光灰柜正面面板与基座。'),
('柜体','右图缘截断灰柜面板、框架、基座及邻接地面。'),
('混合结构','主要是网格地面和灰柜底座边缘，仅少量柜体。'),
('杆体','右图缘极窄斜杆、横件和天空。'),
('杆体','同图右缘杆上部窄框，非中间灰柜。'),
('杆体','同图右缘杆体长条、天空及极少地面。'),
('柜体','右缘截断灰柜宽背面、基座及下方地面。'),
('柜体','灰柜宽侧、右面板和基座。'),
('柜体','蓝柜正面面板、右侧和底座。'),
('柜体','蓝柜宽侧、窄面板、顶部和底座。'),
('柜体','灰柜宽侧、右面板与黑基座。'),
('柜体','暗光灰柜侧面、面板与基座。'),
('柜体','灰柜正面深色面板及基座，顶部部分未包入框。'),
]
REACT = [
('foreground_occlusion','not_truncated','远处圆柱上部露出，下部与基座被前景蓝柜遮挡，细节有限。'),
('none_apparent','not_truncated','近景圆柱顶面、宽主体和方形基座清楚，无明显遮挡。'),
('none_apparent','right_edge_truncated','右缘截断圆柱，但宽弧面、顶面和部分基座仍明显。'),
('none_apparent','not_truncated','与 R02 同图圆柱，主体和基座清楚；分别核对 seed17。'),
]

def main():
    dest=OUT/'review.json'
    if dest.exists(): verify_tree(dest); return
    mp=OUT/'manifest.json'; verify_tree(mp); m=read(mp)
    if len(m['negative'])!=len(OBS) or len(m['reactor_losses'])!=len(REACT): raise ValueError('Scope changed')
    ds=[]; by_image=defaultdict(list); structures=defaultdict(list)
    for i,e in enumerate(m['negative']+m['reactor_losses']):
        d=dict(event_id=e['event_id'],source_event_identity=object_sha256(e),image_sha256=e['image_sha256'],
            evidence_sha256=e['evidence_sha256'],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),status='diagnosed')
        if i<len(OBS):
            d['content_category'],d['reason']=OBS[i]
            by_image[e['image_sha256']].append(e['event_id'])
            # Explicit visual grouping within a camera view, not certified simulator identity.
            group='gray_cabinet' if i+1 in (15,16,17,18,19,20,22,23,26,27,29,30,31,36,37,40,41,42) else d['content_category']
            structures[e['view_id']+':'+group].append(e['event_id'])
        else:
            oc,tr,reason=REACT[i-len(OBS)]
            d.update(content_category='圆柱主体或受遮挡圆柱',reason=reason,occlusion=oc,truncation=tr,
                component_identity='unknown',pixel_visibility_certified=False,operational_misses=e['misses'])
        ds.append(d)
    validate_decisions(dict(negative=m['negative'],reactors=m['reactor_losses']),ds)
    replay={}
    for seed in (7,17,27):
        a=read(SOURCE/'training'/f'evaluation-staged-450-{seed}.json')
        b=read(PRIOR/f'evaluation-retained_appearance-450-{seed}.json')
        replay[str(seed)]=all(a[k]==b[k] for k in ('rows','negative_rows'))
    if not all(replay.values()): raise ValueError('Staged reproducibility changed')
    modules=['tests.test_order_retention','tests.test_order_finalization','tests.test_order_protocol_repair',
        'tests.test_retention450_adapter','tests.test_retention_real_quota','tests.test_retention_regression_review','tests.test_canonical_shutdown']
    tests=subprocess.run([sys.executable,'-m','unittest',*modules],cwd=ROOT,capture_output=True,text=True,check=True)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    inputs={str(mp):file_sha256(mp),str(Path(__file__)):file_sha256(Path(__file__))}
    for mod in modules:
        p=ROOT/(mod.replace('.','/')+'.py');inputs[str(p)]=file_sha256(p)
    frozen(dest,dict(status='review_complete_models_rejected',decisions=ds,selected_family=None,
        negative_content_counts=dict(Counter(d['content_category'] for d in ds[:42])),
        same_image_events=dict(by_image),same_view_visual_structure_events=dict(structures),
        structure_grouping_is_visual_not_simulator_certification=True,
        staged_prediction_reproduction=replay,tests_output=tests.stdout+tests.stderr,baseline=baseline,
        scope='All 42 negative predictions and 4 interleaved reactor losses versus retained_reference-450; not exhaustive visual review of every positive miss.',inputs=inputs))
    print('REVIEW_COMPLETE_MODELS_REJECTED',dict(Counter(d['content_category'] for d in ds[:42])),flush=True)

if __name__=='__main__':main()
