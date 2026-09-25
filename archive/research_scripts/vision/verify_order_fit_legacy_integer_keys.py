"""Narrow compatibility evidence, never a global identity bypass."""
from copy import deepcopy
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def verify_identity(record, kind):
    value = deepcopy(record)
    expected = value.pop('identity')
    if kind == 'hold_frames':
        rows, field = value['frames'], 'instance_mapping'
    elif kind == 'mask_counts':
        rows, field = value['records'], 'visible_pixels_by_label'
    else:
        raise ValueError('Unsupported legacy schema')
    for row in rows:
        mapping = row[field]
        if any(not isinstance(k, str) or not k.isdecimal() or str(int(k)) != k for k in mapping):
            raise ValueError('Noncanonical numeric key')
        row[field] = {int(k): v for k, v in mapping.items()}
    if prior.verify.__globals__['object_sha256'](value) != expected:
        raise ValueError('Legacy identity mismatch')


def run():
    root = prior.ROOT/'data/research/ml_training_recovery_v1'
    targets = [(root/'reviewed-hold-compensation-control-v1/protocol.json', 'hold_frames'),
               (root/'material-view-candidates-v1/N05-expansion-v1/coverage-audit/audit.json', 'mask_counts')]
    rows = []; deps = [Path(__file__).resolve()]
    for path, kind in targets:
        record = prior.read(path); verify_identity(record, kind)
        failures = []
        for filename, digest in record.get('inputs', {}).items():
            try:
                if prior.file_sha256(filename) != digest: raise ValueError('Input hash changed')
            except Exception as exc:
                failures.append(dict(path=filename, reason=str(exc)))
        rows.append(dict(path=str(path), original_identity=record['identity'], schema=kind,
            integer_key_identity_reproduced=True, direct_input_failures=failures,
            checked_direct_inputs=len(record.get('inputs', {})),
            recursive_dependencies_certified=False))
        deps.append(path)
    dest = OUT/'legacy-integer-key-compatibility-v1.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='legacy_identity_explanation_verified_not_dataset_admission',
        records=rows, history_modified=False, dataset_ready=False,
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__ == '__main__':
    for row in run()['records']: print(row['schema'], row['checked_direct_inputs'], row['direct_input_failures'])
