import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src.sandbox.live_replan_gate import validate_record,require_repeat_gate
from src.ml.artifacts import file_sha256

class LiveReplanGateTests(unittest.TestCase):
    def test_missing_campaign_cannot_enable_flight(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OSError):require_repeat_gate(d)

    def test_changed_dependency_and_empty_chain_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.write_text('original')
            receipt=root/'receipt.json';receipt.write_text(json.dumps({'inputs':{str(source):file_sha256(source)}}))
            validate_record(receipt)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Stale'):validate_record(receipt)
            receipt.write_text('{"inputs":{}}')
            with self.assertRaisesRegex(ValueError,'Missing'):validate_record(receipt)

    def test_duplicate_or_incomplete_matrix_rejected(self):
        completion={'status':'repeat_campaign_verified','sandbox_integration_allowed':True,'units':[]}
        protocol={'units':[{'case':'baseline','repeat':1}]*7}
        with patch('src.sandbox.live_replan_gate.validate_record',side_effect=[completion,protocol]):
            with self.assertRaisesRegex(ValueError,'missing or duplicated'):require_repeat_gate('/tmp')

    def test_managed_command_requires_gate_and_disallows_overrides(self):
        from src.sandbox.job_commands import build_command
        config=SimpleNamespace(profile='development',project_root=Path('/tmp'))
        with patch('src.sandbox.live_replan_gate.require_repeat_gate',side_effect=ValueError('blocked')):
            with self.assertRaisesRegex(ValueError,'blocked'):build_command(config,'live-replan-flight')
        with patch('src.sandbox.live_replan_gate.require_repeat_gate',return_value={'campaign_hash':'a'*64}):
            command=build_command(config,'live-replan-flight')
            self.assertIn('--fly',command.argv);self.assertEqual(command.timeout_s,600)
            self.assertEqual(command.budget_paths[0]+'/completion.json',command.expected_outputs[0])
            with self.assertRaises(ValueError):build_command(config,'live-replan-flight',parameters={'speed':1})
            config.profile='formal'
            with self.assertRaises(ValueError):build_command(config,'live-replan-flight')
