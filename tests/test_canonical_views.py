import math
from pathlib import Path
import tempfile
import unittest
from src.vision.canonical.plan import quaternion,rotate,pose_close,visible,write_record,read_record
from src.vision.canonical.collect import paired_after_move,pose_record
from src.vision.canonical.audit import select_new
from src.vision.canonical.cli import add_parsers
import argparse

class CanonicalViewsTests(unittest.TestCase):
    def test_camera_x_forward_rotates_to_north(self):
        v=rotate(quaternion(0,0,math.pi/2),[1,0,0])
        self.assertAlmostEqual(v[0],0);self.assertAlmostEqual(v[1],1)
    def test_positive_pitch_looks_down(self):
        self.assertLess(rotate(quaternion(0,.5,0),[1,0,0])[2],0)
    def test_pose_boundaries(self):
        ref={'position':[0,0,0],'orientation':[0,0,0,1]}
        self.assertTrue(pose_close({'position':[.05,0,0],'orientation':quaternion(0,0,math.radians(1))},ref))
        self.assertFalse(pose_close({'position':[.051,0,0],'orientation':[0,0,0,1]},ref))
        self.assertFalse(pose_close({'position':[0,0,0],'orientation':[0,0,0,0]},ref))
    def test_nan_pose_rejected(self):
        p={'position':[float('nan'),0,0],'orientation':[0,0,0,1]}
        self.assertFalse(pose_close(p,{'position':[0,0,0],'orientation':[0,0,0,1]}))
    def test_old_frame_and_skew(self):
        self.assertFalse(paired_after_move(1,1,1,1,1))
        self.assertTrue(paired_after_move(2,2.033334,2,2,1))
        self.assertFalse(paired_after_move(2,2.033335,2,2,1))
    def test_obstacle_blocks_line(self):
        o=[{'name':'wall','bounds':[1,2,-1,1,0,3]}]
        self.assertFalse(visible([0,0,1],[3,0,1],o,'target'))
        self.assertTrue(visible([0,0,4],[3,0,4],o,'target'))
    def test_exact_and_near_references(self):
        rows=[{'view_id':'one','image_sha256':'a','perceptual_hash':'0000'}, {'view_id':'two','image_sha256':'b','perceptual_hash':'0001'}, {'view_id':'three','image_sha256':'c','perceptual_hash':'ffff'}]
        new,rejected=select_new(rows,[],[rows[2]])
        self.assertEqual([r['view_id'] for r in new],['one'])
        self.assertEqual({r['reason'] for r in rejected},{'within_batch_duplicate','protected_partition_duplicate'})
    def test_repeat_run_is_not_incremental(self):
        r={'view_id':'same','image_sha256':'a','perceptual_hash':'abcd'}
        self.assertEqual(select_new([r],[r],[])[0],[])
    def test_identity_is_order_independent(self):
        with tempfile.TemporaryDirectory() as d:
            a=write_record(Path(d)/'a.json',{'a':1,'b':2});b=write_record(Path(d)/'b.json',{'b':2,'a':1})
            self.assertEqual(a['identity'],b['identity']);self.assertEqual(read_record(Path(d)/'a.json'),a)
    def test_identity_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';p.write_text('{"identity":"bad","x":1}')
            with self.assertRaises(ValueError):read_record(p)
    def test_cli_defaults_calibration(self):
        parser=argparse.ArgumentParser();add_parsers(parser.add_subparsers(dest='command'))
        args=parser.parse_args(['canonical-view-collect','--plan','a','--output','b'])
        self.assertEqual(args.mode,'calibration')
