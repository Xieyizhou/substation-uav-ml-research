"""Reuse exact negative decisions with complete empty-label and RGB checks."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior
from scripts.vision.structure_fit import OUT as HISTORICAL
from scripts.vision.hard_negative_coverage import OUT as COVERAGE
from scripts.vision.hard_negative_coverage_admission import validate_decisions
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE as BRIDGE
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def run():
    p=freeze();hp=HISTORICAL/'protocol.json';old=prior.read(hp);prior.verify(old)
    ap=COVERAGE/'final-admission.json';a=prior.read(ap);prior.verify(a);validate_decisions(a['frames'],a['decisions'])
    sources={d['view_id']:d for d in a['decisions']};deps=[OUT/'protocol.json',hp,ap,Path(__file__).resolve()]
    for stage in ('pilot-v1','remaining-v1'):
        path=BRIDGE/stage/'review-v1/semantic-review.json';r=prior.read(path);prior.verify(r);deps.append(path)
        for d in r['frames']:
            if d['view_id'] in sources:raise ValueError('Duplicate negative source')
            if d['decision']!='accepted' or d['target_exclusion_status']!='no_target_visible_reviewed' or d['truth_object_count']!=0:
                raise ValueError('Unapproved negative source')
            if not d['reason'] or not d['reviewed_at']:raise ValueError('Incomplete review')
            sources[d['view_id']]=d
    oldrows={x['member']['member_id']:x for x in old['negative']};links=[]
    for m in p['members']:
        if m['subset']!='hard_negative':continue
        original=oldrows[m['member_id']];s=original['source'];decision=sources[s['view_id']]
        if m['truth'] or Path(m['label_path']).read_text().strip():raise ValueError('Nonempty negative supervision')
        if prior.file_sha256(s['image_path'])!=s['image_sha256'] or decision['image_sha256']!=s['image_sha256']:
            raise ValueError('Stale negative source review')
        if pixel_hash(m['image_path'])!=pixel_hash(s['image_path']):raise ValueError('Negative source RGB differs')
        deps.append(Path(s['image_path']))
        links.append(dict(member_id=m['member_id'],source_view_id=s['view_id'],source_image=s['image_path'],
            image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],decision=decision,
            status='existing_negative_review_and_empty_full_labels_verified',
            derivation_group=s.get('derivation_group',m['lineage_id']),independent_scene_claim=False))
    dest=OUT/'negative-review-links.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='120_negative_review_links_verified_not_whole_pool_admission',members=links,
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    r=run();print(r['status'],len(r['members']))
