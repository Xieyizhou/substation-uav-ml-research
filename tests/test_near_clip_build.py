import copy
import unittest
from scripts.vision.finalize_near_clip_build_test import compare,OUT,read
from scripts.vision.run_near_clip_build_test import box_map


class ClipBuildTests(unittest.TestCase):
    def setUp(self):
        self.records=[read(OUT/'prefix-isolated'/v/'replay/T027/attempt-01/receipt.json') for v in ('control','clipped')]

    def test_actual_comparison(self):
        self.assertEqual(len(compare(*self.records)),3)

    def test_missing_record(self):
        self.records[1]['records'].pop()
        with self.assertRaises(ValueError):compare(*self.records)

    def test_rgb_mismatch(self):
        self.records[1]['records'][0]['rgb_exact']=False
        with self.assertRaises(ValueError):compare(*self.records)

    def test_existing_box_drift(self):
        self.records[1]['records'][0]['full_boxes']['113'][0]+=2
        with self.assertRaises(ValueError):compare(*self.records)

    def test_missing_recovered_instance(self):
        del self.records[1]['records'][0]['full_boxes']['128']
        with self.assertRaises(ValueError):compare(*self.records)

    def test_cleanup_failure(self):
        self.records[1]['process_cleanup_complete']=False
        with self.assertRaises(ValueError):compare(*self.records)

    def test_duplicate_box_rejected(self):
        message=read(OUT/'prefix-isolated/control/replay/T027/attempt-01/frame-2-boxes.json')
        mapping=read(OUT/'protocol.json')['frames'][0]['instance_mapping']
        message['annotatedBox'].append(copy.deepcopy(message['annotatedBox'][0]))
        with self.assertRaises(ValueError):box_map(message,mapping)


if __name__=='__main__':unittest.main()
