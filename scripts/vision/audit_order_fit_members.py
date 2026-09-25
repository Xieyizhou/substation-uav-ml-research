"""Read-only member/source census; missing evidence never becomes admission."""
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def run():
    p=freeze(); dp=OUT.parent/'design.json'; d=prior.read(dp);prior.verify(d)
    rows=[]; pixels=defaultdict(list); deps=[OUT/'protocol.json',dp,Path(__file__).resolve()]
    for m in p['members']:
        ip=Path(m['image_path']); px=pixel_hash(ip);pixels[px].append(m['member_id'])
        source=m.get('source_image_path',m.get('source_image'))
        source_status='missing'; source_hash=None
        if isinstance(source,str) and Path(source).is_file():
            source_hash=prior.file_sha256(source);deps.append(Path(source))
            source_status='pixel_equal' if pixel_hash(source)==px else 'pixel_mismatch'
        refs={k:m[k] for k in ('source_receipt','source_plan','source_world','evidence_path') if m.get(k)}
        refs_status={}
        for k,v in refs.items():
            path=Path(v)
            refs_status[k]='present' if path.is_file() else 'missing'
            if path.is_file():deps.append(path)
        exposure={key:counts.get(m['member_id'],0) for key,counts in p['actual_exposures'].items()}
        rows.append(dict(member_id=m['member_id'],subset=m['subset'],variant=m.get('variant'),
            lineage_id=m.get('lineage_id'),lineage_resolution=m.get('lineage_resolution','missing'),
            pixel_sha256=px,image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],
            source_image=source,source_file_sha256=source_hash,source_pixel_status=source_status,
            source_refs=refs,source_ref_status=refs_status,actual_exposures=exposure,
            historical_held=m['member_id'] in d['held_members'],full_class_counts=m['class_instances'],
            quality_admission='not_inferred_from_fit_or_file_presence'))
    dest=OUT/'source-census.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='source_census_complete_quality_review_linking_pending',members=rows,
        source_pixel_counts=dict(Counter(r['source_pixel_status'] for r in rows)),
        duplicate_pixel_groups=[v for v in pixels.values() if len(v)>1],
        lineage_resolution_counts=dict(Counter(r['lineage_resolution'] for r in rows)),
        protected_labels_read=False,inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    r=run();print(r['status'],r['source_pixel_counts'],'duplicate groups',len(r['duplicate_pixel_groups']))
