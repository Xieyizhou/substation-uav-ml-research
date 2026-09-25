"""Parent comparisons must follow the frozen initialization, not a global v1."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.ml.artifacts import file_sha256
from src.sandbox.workbench_comparison import compare_with_baseline


class ParentComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.view = self.root / "view"
        self.view.mkdir()
        (self.view / "membership.json").write_text('[{"sample_id":"held-fixed"}]')
        self.weights = self.root / "pinned.pt"
        self.weights.write_bytes(b"actual parent checkpoint")
        self.recipe = SimpleNamespace(parameters={"imgsz":320}, baseline_package_identity_sha256=None,
            pretrained_weights_sha256=file_sha256(self.weights), initialization={
                "checkpoint_path":"pinned.pt", "parent_experiment_id":"parent",
                "parent_receipt_identity_sha256":"a"*64, "parent_threshold":.42})
        self.replay = {"frame_count":1,"threshold":.3,"metrics":{"macro_f1":.8,
            "small_object_recall":.7,"no_target_false_positive_rate":.2}}

    def compare(self, frames=None):
        with patch('src.sandbox.workbench_comparison.collect_predictions', return_value=[{}] if frames is None else frames) as predict, \
             patch('src.sandbox.workbench_comparison.threshold_metrics', return_value={
                 "macro_f1":.7,"small_object_recall":.8,"no_target_false_positive_rate":.1}) as metrics:
            result=compare_with_baseline(self.root,self.view,self.root,self.recipe,self.replay)
        return result,predict,metrics

    def test_actual_pinned_parent_and_frozen_threshold(self):
        result,predict,metrics=self.compare()
        self.assertEqual(predict.call_args.args[0],self.weights.resolve())
        self.assertEqual(predict.call_args.args[1:],(self.view,"validation"))
        self.assertEqual(metrics.call_args.args[1],.42)
        self.assertEqual(result['baseline_experiment_id'],'parent')
        self.assertEqual(result['baseline_model_sha256'],file_sha256(self.weights))
        self.assertEqual(result['membership_sha256'],file_sha256(self.view/'membership.json'))
        self.assertAlmostEqual(result['deltas']['macro_f1'],.1)
        self.assertAlmostEqual(result['deltas']['small_object_recall'],-.1)
        self.assertIn('not held-out',result['scope'])
        self.assertEqual(json.loads((self.root/'comparison.json').read_text()),result)

    def test_changed_parent_snapshot_is_rejected(self):
        self.weights.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'pretrained weights changed'):
            self.compare()

    def test_frame_count_mismatch_cannot_look_like_an_improvement(self):
        with self.assertRaisesRegex(ValueError,'frame count differs'):
            self.compare(frames=[])

    def test_legacy_parent_receipt_is_checked_before_using_its_threshold(self):
        self.recipe.initialization.pop('parent_threshold')
        with patch('src.sandbox.workbench_inference.verified_model',return_value={
            'receipt_identity_sha256':'b'*64,'best_weights_sha256':file_sha256(self.weights),'threshold':.6}):
            with self.assertRaisesRegex(ValueError,'parent identity changed'):
                self.compare()

    def test_external_initialization_declares_validation_selected_threshold(self):
        self.recipe.initialization={'checkpoint_path':'pinned.pt'}
        with patch('src.sandbox.workbench_comparison.select_confidence_threshold',return_value={'selected':{'threshold':.6}}):
            result,_,metrics=self.compare()
        self.assertEqual(result['threshold_policy'],'selection_validation')
        self.assertEqual(metrics.call_args.args[1],.6)
        self.assertEqual(result['baseline_kind'],'initial_weights')
