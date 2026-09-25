"""Read-only recursive check of signed review dependency records."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def walk(paths):
    checked = set(); records = set(); failures = []
    pending = list(paths)
    while pending:
        path = Path(pending.pop()).resolve()
        if str(path) in checked: continue
        checked.add(str(path))
        if not path.exists():
            failures.append(dict(path=str(path), reason='missing')); continue
        if path.suffix != '.json': continue
        try:
            record = prior.read(path)
            if not isinstance(record, dict) or 'identity' not in record: continue
            # Other historical record formats are not silently certified.
            if 'inputs' not in record:
                continue
            records.add(str(path))
            prior.verify(record)
            pending.extend(record['inputs'])
        except Exception as exc:
            failures.append(dict(path=str(path), reason=str(exc)))
    return dict(checked_paths=len(checked), verified_or_attempted_signed_records=len(records), failures=failures)


def run():
    roots = [OUT/'source-spatial-context'/name for name in
             ['background-review-01.json', 'spatial-review-02.json', 'bridge-review-03.json']]
    result = walk(roots)
    dest = OUT/'review-dependency-audit-v1.json'
    if dest.exists():
        old = prior.read(dest); prior.verify(old); return old
    return prior.frozen(dest, dict(status='dependency_check_passed' if not result['failures'] else 'dependency_gaps_found',
        **result, scope='Recursive declared inputs of three spatial review roots; not undeclared dependencies or dataset admission.',
        dataset_ready=False, inputs={str(p):prior.file_sha256(p) for p in roots+[Path(__file__).resolve()]}))


if __name__ == '__main__': print(run())
