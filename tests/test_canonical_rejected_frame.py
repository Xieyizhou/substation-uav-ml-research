import argparse
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.vision.canonical.collect import collect, save_rejected_frame
from src.vision.canonical.cli import add_parsers
from src.ml.artifacts import file_sha256


class RejectedFrameTests(unittest.TestCase):
    def test_rejected_payload_survives_staging_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with tempfile.TemporaryDirectory() as staging:
                stage=Path(staging)
                (stage/'frame.ppm').write_bytes(b'P6\n1 1\n255\n\x00\x01\x02')
                (stage/'frame.depth').write_bytes(b'\x00\x00\x80\x3f')
                rgb=SimpleNamespace(payload_relative_path='frame.ppm',capture_timestamp=1.0,width=1,height=1)
                depth=SimpleNamespace(payload_relative_path='frame.depth',capture_timestamp=1.001,width=1,height=1)
                row=save_rejected_frame(stage,root,'view',rgb,depth)
            self.assertEqual(file_sha256(row['rgb_path']),row['image_sha256'])
            self.assertEqual(file_sha256(row['depth_path']),row['depth_sha256'])
            self.assertFalse(row['training_admitted'])

    def test_single_view_cannot_bypass_pilot_review(self):
        with patch('src.vision.canonical.collect.read_record',return_value={'calibration_views':[{'view_id':'one'}]}):
            with self.assertRaises(ValueError):
                asyncio.run(collect('plan.json','unused',mode='pilot',view_id='one'))
            with self.assertRaises(ValueError):
                asyncio.run(collect('plan.json','unused',view_id='unknown'))

    def test_cli_single_view(self):
        parser=argparse.ArgumentParser()
        add_parsers(parser.add_subparsers(dest='command'))
        args=parser.parse_args(['canonical-view-collect','--plan','a','--output','b','--view-id','one'])
        self.assertEqual(args.view_id,'one')
        self.assertEqual(args.mode,'calibration')
