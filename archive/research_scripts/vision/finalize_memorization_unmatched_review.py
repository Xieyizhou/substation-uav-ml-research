"""Finalize visual attribution of unmatched training-image predictions."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


def main():
    base = ROOT / 'data/research/ml_training_recovery_v1/memorization-v1/unmatched-audit-v1'
    source = base / 'report.json'
    report = json.loads(source.read_text())
    # The four target-overlap rows are clearly class-confused in the contact sheets;
    # the remaining orange boxes lie in empty sky/background regions.
    wrong_class = {1, 3, 15, 18}
    background = {2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19}
    assert wrong_class | background == set(range(1, 20))
    assert not wrong_class & background
    rows = []
    for item in report['queue']:
        idx = item['review_index']
        category = 'confirmed_wrong_class_on_visible_target' if idx in wrong_class else 'confirmed_background_hallucination'
        rows.append({**item, 'visual_category': category,
                     'visual_review_note': 'Orange box overlaps a green target but class is wrong.' if idx in wrong_class
                     else 'Orange box is in empty sky/background and does not overlap any labeled object.'})
    finalized = {**report, 'queue': rows,
                 'visual_review': {'reviewer': 'assistant',
                                   'reviewed_pages': sorted(report['visuals']),
                                   'method': 'All four contact sheets reviewed at 640x360 with every truth box overlaid; this is an assistant visual review, not a blinded independent annotation.',
                                   'confirmed_wrong_class_count': len(wrong_class),
                                   'confirmed_background_hallucination_count': len(background),
                                   'geometric_categories_are_not_used_as_final_labels': True},
                 'status': 'visual_review_complete_training_only',
                 'training_set_only': True, 'promotable': False,
                 'limits': ['The 19 unmatched predictions are on training images and cannot estimate generalization.',
                            'This review classifies the predictions, not the causes of the underlying model behavior.',
                            'The two missed truth instances remain missed at this matching threshold.',
                            'No cabinet region was added as a target label.']}
    finalized.pop('identity', None)
    finalized['identity'] = object_sha256(finalized)
    write_json(base / 'final-review.json', finalized)
    print(json.dumps({'wrong_class': len(wrong_class), 'background': len(background),
                      'missed_truth_instances': len(report['missed_targets']),
                      'identity': finalized['identity']}, indent=2))


if __name__ == '__main__':
    main()
