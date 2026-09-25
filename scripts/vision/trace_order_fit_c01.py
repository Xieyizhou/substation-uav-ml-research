"""Post-hoc C01 source correspondence, not whole-scene approval."""
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision.trace_order_fit_legacy_worlds import resolve_labels, model_mapping, equal_rgb


def run():
    pp = OUT/'protocol.json'
    cp = prior.ROOT/'data/research/ml_training_recovery_v1/material-view-candidates-v1/compensating-pose-pilot-v1/real-verification-v1/C01/protocol.json'
    p, capture = prior.read(pp), prior.read(cp)
    prior.verify(p); prior.verify(capture)
    m = next(x for x in p['members'] if x['member_id'] == 'C01-original'); f = capture['frame']
    receipt = prior.read(f['source_receipt']); prior.verify(receipt)
    plan = prior.read(f['source_plan'])
    if prior.file_sha256(f['source_world']) != plan['files']['world.sdf']: raise ValueError('World changed')
    equal_rgb(f['source_image'], m['image_path'])
    mapping, _ = model_mapping(ET.parse(f['source_world']))
    labels = resolve_labels(m['truth'], receipt['truth']['objects'], mapping)
    for key, value in f['instance_mapping'].items():
        if mapping[str(key)] != value['object_id']: raise ValueError('Mapping mismatch')
    dest = OUT/'c01-source-correspondence.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    deps = [pp, cp, Path(__file__).resolve(), Path(m['image_path']), Path(m['label_path'])]
    deps.extend(Path(f[k]) for k in ['source_image','source_receipt','source_plan','source_world'])
    return prior.frozen(dest, dict(status='C01_original_rgb_and_eleven_label_instances_correspond',
        member_id=m['member_id'], annotations=labels, source_frame=f,
        training_eligible=False, pixel_visibility_certified=False,
        remaining='Central unboxed blue cabinet and lower-edge building require separate source/spatial association.',
        inputs={str(path):prior.file_sha256(path) for path in deps}))


if __name__ == '__main__': print(run()['status'])
