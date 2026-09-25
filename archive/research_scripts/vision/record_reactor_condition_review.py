"""Explicit visual-condition review for all current reactor inventory pages."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.build_scale_reactor_coverage import OUT, prior
from scripts.vision.record_reviewed_endpoint_review import validate

# Every page was inspected.  States describe content/occlusion, not asset identity
# and do not certify pixel-level visibility.
OVERRIDES = {
    **{f'R{i:03}': ('small_clear', '主体圆柱和方形基座可辨，视角较远但未见前景覆盖。') for i in (1,69,70,71,72)},
    **{f'R{i:03}': ('clear_body', '主体圆柱、椭圆顶面和方形基座清楚；邻近设备未覆盖目标主体。') for i in range(2,35)},
    **{f'R{i:03}': ('partial_occlusion', '圆柱主体大部可辨，但前景块体／基座边缘覆盖目标局部，不能作完全无遮挡证据。') for i in (35,36,37)},
    'R040': ('partial_occlusion', '主体圆柱、顶面和基座可辨，近处电容器顶面叠入目标框下方。'),
    **{f'R{i:03}': ('clear_body', '主体圆柱、顶面和基座清楚，材质变化未改变几何与可辨结构。') for i in range(38,69)},
}
# Specific variant labels are visual rendering conditions; they are not asset claims.

def run():
    inv = OUT/'inventory.json'; r = prior.read(inv); prior.verify(r)
    expected = [x['review_id'] for x in r['rows']]
    if set(expected) != set(OVERRIDES) or len(expected) != 72:
        raise ValueError('Inventory population changed')
    now = datetime.now(timezone.utc).isoformat(); decisions=[]
    for x in r['rows']:
        state,reason=OVERRIDES[x['review_id']]
        paths=[inv,Path(x['page']),Path(x['image_path']),Path(x['label_path'])]
        decisions.append(dict(review_id=x['review_id'],member_id=x['member_id'],subset=x['subset'],variant=x['variant'],
            lineage_id=x['lineage_id'],class_name='reactor',truth=x['truth'],short_side_640=x['short_side_640'],
            visual_state=state,content_evidence=['body','base'],reason=reason,
            review_nature='AI辅助审核',reviewed_at=now,pixel_visibility_certified=False,
            simulation_asset_identity='source_identity_from_manifest_not_visual_inference',
            actual_image_exposures=x['actual_image_exposures'],training_admitted=False,promotable=False,
            evidence_hashes={str(p):prior.file_sha256(p) for p in paths}))
    validate(decisions,expected,'review_id')
    by_state=Counter(x['visual_state'] for x in decisions)
    by_variant=defaultdict(Counter)
    for x in decisions: by_variant[x['variant']][x['visual_state']]+=1
    by_subset=defaultdict(Counter)
    for x in decisions: by_subset[x['subset']][x['visual_state']]+=1
    dest=OUT/'explicit-review.json'
    if dest.exists():
        old=prior.read(dest);prior.verify(old);validate(old['decisions'],expected,'review_id');return old
    return prior.frozen(dest,dict(status='reactor_condition_review_complete_with_limits',decisions=decisions,
        total=len(decisions),state_counts=dict(by_state),state_by_variant={k:dict(v) for k,v in by_variant.items()},
        state_by_subset={k:dict(v) for k,v in by_subset.items()},
        note='AI-assisted visual review of displayed pages; no instance mask and no pixel-level visibility certification. Shared lineages and variants are not independent scenes.',
        training_admitted=False,promotable=False,
        inputs={str(p):prior.file_sha256(p) for p in (inv,Path(__file__).resolve(),Path(__file__).with_name('build_scale_reactor_coverage.py'))}))

if __name__=='__main__':
    x=run();print(x['status']);print(x['state_counts']);print(x['state_by_variant'])
