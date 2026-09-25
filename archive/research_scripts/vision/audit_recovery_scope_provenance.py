"""Inventory development receipt evidence without inferring historical modes."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


def mode_fields(value, prefix=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f'{prefix}.{key}'
            if key in ('box_type', 'annotation_mode', 'bbox_coordinate_convention'):
                found.append({'field': path, 'value': child})
            found.extend(mode_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(mode_fields(child, f'{prefix}[{index}]'))
    return found


def main():
    base = ROOT / 'data/research/ml_training_recovery_v1'
    source = base / 'annotation-compatibility-audit-v1/report.json'
    inputs = {str(source): file_sha256(source), __file__: file_sha256(Path(__file__))}
    rows = []
    for item in json.loads(source.read_text())['legacy_collection_provenance']:
        folder = Path(item['collection'])
        evidence = []
        for path in (folder / 'collection-receipt.json', folder.parent / 'run-receipt.json', folder.parent / 'prearm.json'):
            if not path.exists():
                continue
            inputs[str(path)] = file_sha256(path)
            record = json.loads(path.read_text())
            evidence.append({'path': str(path), 'mode_fields': mode_fields(record)})
        receipt = json.loads((folder / 'collection-receipt.json').read_text())
        truth = folder / receipt['truth']['relative_path']
        inputs[str(truth)] = file_sha256(truth)
        if inputs[str(truth)] != receipt['truth']['sha256']:
            raise ValueError(f'Changed raw truth: {truth}')
        fields = Counter()
        count = 0
        for line in truth.read_text().splitlines():
            if line.strip():
                for field in mode_fields(json.loads(line)):
                    fields[(field['field'].split('.')[-1], str(field['value']))] += 1
                count += 1
        rows.append({'collection': str(folder), 'receipt_evidence': evidence,
                     'raw_truth_hash_verified': True, 'raw_truth_rows': count,
                     'raw_truth_mode_fields': [{'field': k[0], 'value': k[1], 'count': v} for k, v in sorted(fields.items())],
                     'historical_sensor_snapshot_verified': False,
                     'disposition': 'keep_pending_scope_provenance'})
    out = base / 'scope-provenance-v1'
    out.mkdir(exist_ok=True)
    report = {'schema_version': 1, 'inputs': inputs, 'collections': rows,
              'collection_count': len(rows),
              'collections_with_coordinate_convention_fields': sum(any(f['field'] == 'bbox_coordinate_convention' for f in r['raw_truth_mode_fields']) for r in rows),
              'collections_with_explicit_sensor_mode': sum(any(f['field'].split('.')[-1] in ('annotation_mode', 'box_type') for f in r['raw_truth_mode_fields'] + [f for e in r['receipt_evidence'] for f in e['mode_fields']]) for r in rows),
              'training_admitted': False,
              'limits': ['Development candidate source receipts and raw truth only; no protected labels inspected.',
                         'Receipt fields alone do not certify the simulator-loaded sensor snapshot.',
                         'Current source code and fractional coordinates do not prove historical mode.',
                         'No replay or historical evaluation scope certification performed.']}
    report['identity'] = object_sha256(report)
    write_json(out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('inputs', 'collections')}, indent=2))


if __name__ == '__main__':
    main()
