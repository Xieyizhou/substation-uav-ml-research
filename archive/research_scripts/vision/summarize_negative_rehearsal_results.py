"""Summarize existing predictions with undefined precision retained."""
from pathlib import Path
from src.vision.canonical.plan import read_record, write_record
from src.ml.artifacts import file_sha256
from scripts.vision.prepare_negative_rehearsal_control import OUT, SOURCE, KEYS
from scripts.vision.finalize_exposure_diagnosis import aggregate


def main():
    paths = [OUT/'evaluation-v1'/f'{k}.json' for k in KEYS]
    oldpaths = [SOURCE/'evaluation-v2'/f'visibility-452-{s}.json' for s in (7,17,27)]
    records = [read_record(p) for p in paths + oldpaths]
    for r in records:
        for p, h in r['inputs'].items():
            if file_sha256(p) != h: raise ValueError('Stale input: '+p)
    current, previous = aggregate(records[:3]), aggregate(records[3:])
    prefixes = {}
    for s, key in zip((7,17,27), KEYS):
        cp = OUT/'training'/key/'completion.json'
        op = SOURCE/'training'/f'visibility-452-{s}'/'completion.json'
        c, o = read_record(cp), read_record(op)
        prefixes[str(s)] = [r['loss_items'] for r in c['loss_curve'][:460]] == [r['loss_items'] for r in o['loss_curve']]
        paths += [cp, op]
    dest = OUT/'evaluation-v1'/'comparison.json'
    if not dest.exists():
        write_record(dest, dict(status='numerical_complete_failed_detection_gates_visual_review_pending',
            current=current, previous=previous, first_460_step_loss_exactly_equal=prefixes,
            matching_conflicts=sum(r['matching_conflicts'] for r in records[:3]),
            selected_candidate=None, training_admitted=False, promotable=False,
            conclusion='Original planned hit and full-image recall are zero for all three seeds. Lower negative FPR is accompanied by severe detection loss.',
            next_direction='Diagnose the negative-only tail, including confidence and batch-normalization state; any further training should use a frozen controlled design.',
            inputs={str(p):file_sha256(p) for p in paths+oldpaths+[Path(__file__).resolve()]}))
    print(dest)


if __name__ == '__main__': main()
