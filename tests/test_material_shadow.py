import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
from dataclasses import dataclass
import json
import time
import numpy as np
from scripts.vision import material_shadow as s

class ShadowTests(unittest.TestCase):
    def test_warmup_is_synthetic_inference_only(self):
        model=Mock();model.predict.return_value=[]
        r=s.warmup(model)
        self.assertEqual(model.predict.call_count,3)
        self.assertTrue(r['synthetic_predictions_discarded'])
        self.assertEqual(len(r['call_ms']),3)
        model.train.assert_not_called()
        for call in model.predict.call_args_list:
            self.assertEqual(call.kwargs['source'].shape,(640,640,3))
            self.assertFalse(call.kwargs['source'].any())

    def test_rgb_and_protocol(self):
        model=Mock();model.predict.return_value=[]
        result=s.infer(model,np.array([[[1,2,3]]],dtype=np.uint8))
        np.testing.assert_array_equal(model.predict.call_args.kwargs['source'],[[[3,2,1]]])
        for k,v in s.PROTOCOL.items():self.assertEqual(model.predict.call_args.kwargs[k],v)
        self.assertEqual(result['control_authority'],'none')
        self.assertIs(model.predict.call_args.kwargs['rect'],False)

    def test_latest_queue(self):
        q=asyncio.Queue(maxsize=1)
        self.assertEqual(s.latest_put(q,1),0)
        self.assertEqual(s.latest_put(q,2),1)
        self.assertEqual(q.get_nowait(),2)

    def test_start_failure_cleanup(self):
        class Source:
            stopped=False
            async def start(self):raise RuntimeError('missing simulator')
            async def stop(self):self.stopped=True
        source=Source()
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(RuntimeError,'missing simulator'):
            asyncio.run(s.observe(None,Path(d),'auto',1,lambda *a,**kw:source))
        self.assertTrue(source.stopped)

    def test_no_control_dependencies(self):
        import ast
        tree=ast.parse(Path(s.__file__).read_text())
        modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        modules += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
        self.assertFalse(any(x.startswith(('mavsdk','src.flight','src.planner')) for x in modules))

    def test_live_timing_partition_and_cleanup(self):
        @dataclass
        class Frame:
            receive_monotonic_timestamp: float
        class Source:
            stopped=False
            async def start(self):pass
            async def stop(self):self.stopped=True
            async def events(self,**kw):
                yield SimpleNamespace(valid=True,frame=Frame(time.monotonic()))
                await asyncio.sleep(10)
        source=Source();decoded=SimpleNamespace(image=SimpleNamespace(array=None,decoded_content_sha256='test'))
        with tempfile.TemporaryDirectory() as d,patch.object(s,'decode_camera_payload',return_value=decoded),patch.object(s,'infer',return_value=dict(predictions=[],inference_call_ms=0,control_authority='none')):
            counts=asyncio.run(s.observe(None,Path(d),'test',.05,lambda *a,**kw:source))
            row=json.loads((Path(d)/'detections.jsonl').read_text())
            parts=[row[f] for f in ('source_encode_dispatch_ms','application_queue_wait_ms','decode_ms','inference_dispatch_and_call_ms')]
            self.assertTrue(all(x>=0 for x in parts))
            self.assertAlmostEqual(sum(parts),row['receive_to_result_ms'],places=5)
            self.assertIsNone(row['capture_to_inference_age_s'])
            self.assertEqual(counts['inferred'],1)
        self.assertTrue(source.stopped)

if __name__=='__main__':unittest.main()
