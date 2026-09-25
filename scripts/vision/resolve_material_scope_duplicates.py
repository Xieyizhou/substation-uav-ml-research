"""Resolve only exact same-source target/all-body aliases; external overlap blocks."""
from pathlib import Path
from scripts.vision.prepare_material_retention_coverage import OUT,prior
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def inspect(m):
    rows={r['member_id']:r for r in m['members']};resolved=[]
    for gap in m['reference_overlap_gaps']:
        a=rows[gap['member_id']]
        if len(gap['overlap'])!=1 or not gap['overlap'][0].startswith('candidate_exact_pixel_duplicate:'):
            raise ValueError('External or ambiguous overlap is not allowed')
        other=gap['overlap'][0].split(':',1)[1];b=rows[other]
        if a['pair_id']!=b['pair_id'] or {a['variant'],b['variant']}!={'gray_target_body','gray_all_body'}:
            raise ValueError('Duplicate is not the frozen same-source contrast')
        if a['full_truth']!=b['full_truth'] or Path(a['label_path']).read_bytes()!=Path(b['label_path']).read_bytes():raise ValueError('Duplicate supervision conflict')
        if pixel_hash(Path(a['image_path']))!=pixel_hash(Path(b['image_path'])):raise ValueError('Duplicate pixels stale')
        resolved.append(dict(members=[other,a['member_id']],pair_id=a['pair_id'],pixel_sha256=a['pixel_sha256'],
            independent_sample_count=1,coverage_contrast_visible_effect=False))
    return resolved


def main():
    ep=OUT/'candidate-export-v1/manifest.json';m=prior.read(ep);prior.verify(m)
    resolved=inspect(m);dest=OUT/'duplicate-resolution.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='same_source_aliases_resolved_no_external_overlap',groups=resolved,
        unique_candidate_pixels=len({r['pixel_sha256'] for r in m['members']}),candidate_rows=len(m['members']),
        interpretation='Keep historical export unchanged; internal duplicates are paired intervention aliases, not independent samples.',
        inputs={str(p):prior.file_sha256(p) for p in (ep,Path(__file__).resolve())}))


if __name__=='__main__':print(main()['status'])
