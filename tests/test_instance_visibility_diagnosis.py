import asyncio
import copy
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scripts.vision.instance_visibility_diagnosis import add_sensor,validate_world,command,stop_group,save,read
from scripts.vision.run_instance_visibility_window import check_mask,certify
from scripts.vision.exposure_protocol import verify
from src.vision.canonical.gates import instance_mapping
from scripts.vision.finalize_instance_visibility import validate_reviews

WORLD='''<sdf><world name="test"><model name="canonical_camera"><link name="research_camera_link"><sensor name="research_rgb" type="camera"><topic>image</topic><camera><horizontal_fov>1.466</horizontal_fov><image><width>1920</width><height>1080</height></image></camera></sensor></link></model></world></sdf>'''

class VisibilityTests(unittest.TestCase):
    def test_world_allowlist(self):
        tree=ET.ElementTree(ET.fromstring(WORLD));derived=add_sensor(tree)
        validate_world(tree,derived)
        derived.find('.//world').set('name','changed')
        with self.assertRaises(ValueError):validate_world(tree,derived)

    def test_sensor_mismatch(self):
        tree=ET.ElementTree(ET.fromstring(WORLD));derived=add_sensor(tree)
        derived.find(".//sensor[@name='diagnostic_instances']/camera/horizontal_fov").text='1.0'
        with self.assertRaises(ValueError):validate_world(tree,derived)

    def test_instance_mapping_collision(self):
        obj=dict(category='switchgear',name='one',runtime_labels=[149])
        with self.assertRaises(ValueError):instance_mapping({'objects':[obj,dict(obj,name='two')]})

    def test_masks_fail_closed(self):
        a=np.zeros((1080,1920,3),dtype='u1');a[0,0]=[1,0,149];m={'pixelFormatType':'RGB_INT8'}
        self.assertEqual(check_mask(m,a,{'149':{}}),[0,149])
        with self.assertRaises(ValueError):check_mask(m,a,{'149':{}},'semantic')
        with self.assertRaises(ValueError):check_mask(m,a,{'113':{}})
        a[0,0]=[0,0,149]
        with self.assertRaises(ValueError):check_mask(m,a,{'149':{}})
        a[:]=0
        with self.assertRaises(ValueError):check_mask(m,a,{'149':{}})

    def test_exact_alignment_required(self):
        self.assertTrue(certify(True,True,3,33.334,1,True))
        for values in [(False,True,3,0,0,True),(True,False,3,0,0,True),(True,True,2,0,0,True),(True,True,3,34,0,True),(True,True,3,0,1.01,True),(True,True,3,0,0,False)]:
            self.assertFalse(certify(*values))

    def test_hash_roundtrip_and_stale(self):
        from scripts.vision.instance_visibility_diagnosis import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source';p.write_text('source');r=Path(folder)/'record.json'
            save(r,dict(instance_mapping={'149':{},'79':{}},inputs={str(p):file_sha256(p)}));verify(read(r))
            p.write_text('changed')
            with self.assertRaises(ValueError):verify(read(r))

    def test_review_missing_duplicate_expired(self):
        from scripts.vision.instance_visibility_diagnosis import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'image';f.write_bytes(b'evidence')
            p={'frames':[{'review_ids':['T049']}]}
            r=dict(review_id='T049',status='evidence_insufficient',review_nature='AI-assisted',reviewed_at='2026-09-08',reason='occluded',inputs={str(f):file_sha256(f)})
            validate_reviews(p,[r])
            for rows in ([],[r,r],[dict(r,status='visible_identifiable')]):
                with self.assertRaises(ValueError):validate_reviews(p,rows)
            f.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_reviews(p,[r])

    def test_empty_mask_and_fragment_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            from scripts.vision.instance_visibility_diagnosis import file_sha256
            f=Path(folder)/'evidence';f.write_bytes(b'mask')
            p={'frames':[{'review_ids':['T049']}]}
            r=dict(review_id='T049',status='no_instance_pixels',review_nature='AI-assisted',reviewed_at='2026-09-08',reason='zero target pixels',original_frame_certified=True,visible_pixel_count=0,inputs={str(f):file_sha256(f)})
            validate_reviews(p,[r])
            with self.assertRaises(ValueError):validate_reviews(p,[dict(r,original_frame_certified=False)])
            with self.assertRaises(ValueError):validate_reviews(p,[dict(r,status='visible_content_insufficient')])
            validate_reviews(p,[dict(r,status='visible_content_insufficient',visible_pixel_count=1,component_evidence='unknown')])

class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_cleanup(self):
        p=await asyncio.create_subprocess_exec(sys.executable,'-c','import time;time.sleep(30)',start_new_session=True,stderr=asyncio.subprocess.DEVNULL)
        await stop_group(p);self.assertIsNotNone(p.returncode)

    async def test_command_timeout_and_cancel(self):
        with self.assertRaises(asyncio.TimeoutError):await command(sys.executable,'-c','import time;time.sleep(30)',timeout=.05)
        task=asyncio.create_task(command(sys.executable,'-c','import time;time.sleep(30)',timeout=30))
        await asyncio.sleep(.05);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
