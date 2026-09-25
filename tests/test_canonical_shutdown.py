import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.vision.canonical.collect import collect


class CanonicalShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_failure_stops_bridge_before_removing_frames(self):
        order = []
        with tempfile.TemporaryDirectory() as directory:
            stage = Mock(name='staging')
            stage.name = directory
            stage.cleanup.side_effect = lambda: order.append('frames_removed')
            source = Mock()
            source.start = AsyncMock(side_effect=RuntimeError('injected bridge startup failure'))
            source.stop = AsyncMock(side_effect=lambda: order.append('bridge_stopped'))
            source.receipt.return_value = {}
            processes = []
            for _ in range(3):
                process = Mock()
                process.pid = 12345
                process.returncode = None
                process.stdout = asyncio.StreamReader()
                process.communicate = AsyncMock(return_value=(b'', b''))
                process.wait = AsyncMock(return_value=0)
                processes.append(process)
            topics = '\n'.join([
                '/research_camera/image', '/research_camera/depth',
                '/research_camera/boxes', '/world/test/pose/info',
                'gz.msgs.Image', 'gz.msgs.AnnotatedAxisAligned2DBox_V',
                'gz.msgs.Pose_V',
            ])
            plan = {'identity':'fixture', 'files':{}, 'world_name':'test',
                    'map_id':'simple', 'calibration_views':[], 'pilot_views':[]}
            with patch('src.vision.canonical.collect.read_record', return_value=plan), \
                 patch('src.vision.canonical.collect.tempfile.TemporaryDirectory', return_value=stage), \
                 patch('src.vision.canonical.collect.NativeGazeboRgbDepthSource', return_value=source), \
                 patch('src.vision.canonical.collect.command', new=AsyncMock(return_value=topics)), \
                 patch('src.vision.canonical.collect.validate_preflight', return_value=({'actual_annotation_mode':'full_2d','world_sha256':'0'}, {}, {})), \
                 patch('src.vision.canonical.collect.asyncio.create_subprocess_exec', new=AsyncMock(side_effect=processes)), \
                 patch('src.vision.canonical.collect.os.killpg'):
                result = await collect(Path(directory)/'plan.json', Path(directory)/'run')
            self.assertEqual(result['status'], 'blocked')
            self.assertIn('injected bridge startup failure', result['error'])
            self.assertEqual(order, ['bridge_stopped', 'frames_removed'])
            source.stop.assert_awaited_once()
            for process in processes[1:]:
                process.communicate.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
