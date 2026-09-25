import copy
import json
import unittest
from unittest.mock import patch

from scripts.vision.finalize_full_box_runtime_trace import OUT,read,validate_trace,validate_alignment
from scripts.vision.verify_full_box_runtime_trace import guard


class RuntimeTraceTests(unittest.TestCase):
    def setUp(self):
        self.frame=read(OUT/'protocol.json')['frames'][0]
        folder=OUT/'replay/T027/attempt-01'
        self.events=[json.loads(s) for s in (folder/'runtime-trace.jsonl').read_text().splitlines()]
        self.receipt=read(folder/'receipt.json')

    def test_valid_runtime_trace(self):
        self.assertEqual(len(validate_trace(self.events,self.frame)),3)
        validate_alignment(self.receipt)

    def test_missing_or_duplicate_event(self):
        for events in (self.events[:-1],self.events+[self.events[-1]]):
            with self.assertRaises(ValueError):validate_trace(events,self.frame)

    def test_wrong_binary(self):
        self.events[0]['uuid']='unknown'
        with self.assertRaises(ValueError):validate_trace(self.events,self.frame)

    def test_unobserved_branch(self):
        self.events[2]['branch']='accepted'
        with self.assertRaises(ValueError):validate_trace(self.events,self.frame)

    def test_wrong_instance(self):
        self.events[1]['label']=127
        with self.assertRaises(ValueError):validate_trace(self.events,self.frame)

    def test_wrong_component(self):
        self.events[1]['position'][2]=0
        with self.assertRaises(ValueError):validate_trace(self.events,self.frame)

    def test_changed_camera(self):
        self.events[1]['view_matrix'][3]+=1
        with self.assertRaises(ValueError):validate_trace(self.events,self.frame)

    def test_bad_projection(self):
        self.events[1]['min_vertex'][0]=-.5
        with self.assertRaises(ValueError):validate_trace(self.events,self.frame)

    def test_rgb_mismatch_or_cleanup_failure(self):
        for key in ('control_rgb_exact','process_cleanup_complete'):
            receipt=copy.deepcopy(self.receipt);receipt[key]=False
            with self.assertRaises(ValueError):validate_alignment(receipt)

    def test_sync_failure(self):
        self.receipt['comparisons'][0]['visible_skew_ms']=40
        with self.assertRaises(ValueError):validate_alignment(self.receipt)

    def test_stale_binary_rejected(self):
        with patch('scripts.vision.verify_full_box_runtime_trace.file_sha256',return_value='changed'):
            with self.assertRaises(ValueError):guard()


if __name__=='__main__':unittest.main()
