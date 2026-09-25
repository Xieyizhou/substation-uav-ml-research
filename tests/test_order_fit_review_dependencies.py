"""Ensure transitive stale evidence cannot hide behind a valid root receipt."""
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.vision.verify_order_fit_review_dependencies import walk


class DependencyTests(unittest.TestCase):
    def test_stale_descendant_is_reported(self):
        root, child = Path('/root.json'), Path('/child.json')
        records = {root: {'identity': 'root', 'inputs': {str(child): 'hash'}},
                   child: {'identity': 'child', 'inputs': {'/evidence.png': 'old'}}}
        def verify(record):
            if record['identity'] == 'child': raise ValueError('Input bytes changed: /evidence.png')
        with patch.object(Path, 'exists', return_value=True), \
             patch('scripts.vision.verify_order_fit_review_dependencies.prior.read', side_effect=lambda p: records[p]), \
             patch('scripts.vision.verify_order_fit_review_dependencies.prior.verify', side_effect=verify):
            result = walk([root])
        self.assertEqual(result['verified_or_attempted_signed_records'], 2)
        self.assertEqual(result['failures'][0]['path'], str(child))

    def test_cycles_and_shared_dependencies_terminate(self):
        root = Path('/root.json')
        record = {'identity': 'root', 'inputs': {str(root): 'hash'}}
        with patch.object(Path, 'exists', return_value=True), \
             patch('scripts.vision.verify_order_fit_review_dependencies.prior.read', return_value=record), \
             patch('scripts.vision.verify_order_fit_review_dependencies.prior.verify'):
            result = walk([root, root])
        self.assertEqual(result['checked_paths'], 1)
        self.assertEqual(result['failures'], [])


if __name__ == '__main__': unittest.main()
