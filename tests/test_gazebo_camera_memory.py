import base64
from pathlib import Path
import tempfile
import unittest
import asyncio
import json
from src.sensors.gazebo_camera_latest import RawCameraEvent
from scripts.vision.material_shadow import latest_put
import numpy as np
from src.sensors.gazebo_image_codec import parse_gazebo_image_message
from src.sensors.gazebo_camera_memory import parse_memory,persist_event
from src.sensors.camera_decoder import decode_camera_payload

class MemoryTests(unittest.TestCase):
    def test_all_formats_padding_and_evidence(self):
        for fmt,channels in [('L_INT8',1),('RGB_INT8',3),('BGR_INT8',3),('RGBA_INT8',4),('BGRA_INT8',4)]:
            with self.subTest(fmt=fmt),tempfile.TemporaryDirectory() as d:
                root=Path(d);width,height=3,2;step=width*channels+2
                msg=dict(width=width,height=height,step=step,pixel_format_type=fmt,data=base64.b64encode(bytes(range(step*height))).decode(),header={'stamp':{'sec':1,'nsec':2}})
                old=parse_gazebo_image_message(msg,topic='/test',staging_directory=root/'png',receive_index=1,received_monotonic=2.)
                decoded=decode_camera_payload(old,root/'png');new=parse_memory(msg,'/test',1,2.)
                np.testing.assert_array_equal(decoded.image.array,new.decoded.image.array)
                self.assertEqual(decoded.image.decoded_content_sha256,new.decoded.image.decoded_content_sha256)
                persist_event(new,root/'raw');persist_event(new,root/'raw')
                again=decode_camera_payload(new.frame,root/'raw')
                np.testing.assert_array_equal(again.image.array,decoded.image.array)
                self.assertEqual(new.frame.capture_timestamp,old.capture_timestamp)
                self.assertEqual(new.frame.sequence_number,old.sequence_number)
                self.assertFalse(new.decoded.image.array.flags.writeable)
    def test_invalid_payload(self):
        for data in ('!','AA=='):
            with self.assertRaises(ValueError):parse_memory(dict(width=3,height=2,pixel_format_type='RGB_INT8',data=data),'/t',1,0.)

    def test_drop_before_decode(self):
        q=asyncio.Queue(maxsize=1)
        latest_put(q,RawCameraEvent(1,b'invalid old JSON','/t',0.))
        msg=dict(width=1,height=1,step=3,pixel_format_type='RGB_INT8',data='AQID',header={'stamp':{'sec':1,'nsec':0}})
        self.assertEqual(latest_put(q,RawCameraEvent(2,json.dumps(msg).encode(),'/t',2.)),1)
        event=q.get_nowait().materialize()
        self.assertEqual(event.frame.receive_monotonic_timestamp,2.)
        np.testing.assert_array_equal(event.decoded.image.array,[[[1,2,3]]])

if __name__=='__main__':unittest.main()
