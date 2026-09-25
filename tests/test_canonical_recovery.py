import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from src.vision.canonical.recovery import acknowledged, acquire_view, resumed_views


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def test_service_requires_explicit_boolean_response(self):
        self.assertTrue(acknowledged('data: true\n'))
        self.assertFalse(acknowledged('request true; data: false'))
        self.assertFalse(acknowledged('true'))

    async def test_retries_missing_ack_then_uses_fresh_attempt(self):
        active = {}
        move = AsyncMock(side_effect=['data: false', 'data: true'])
        async def producer():
            while not active:
                await asyncio.sleep(.001)
            active['result'] = {'view_id': 'a', 'status': 'captured'}
        producer_task = asyncio.create_task(producer())
        result = await acquire_view({'view_id': 'a'}, move=move, active=active,
                                    get_fence=lambda: 5, check_fatal=lambda: None)
        await producer_task
        self.assertEqual(result['status'], 'captured')
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(active, {})

    async def test_timeout_is_bounded_and_records_diagnostics(self):
        active = {}
        def health():
            active['diagnostics']['pose_outside_tolerance'] = 7
        result = await acquire_view({'view_id': 'a'}, move=AsyncMock(return_value='data: true'),
                                    active=active, get_fence=lambda: 1, check_fatal=health,
                                    timeout_s=.001, max_attempts=2)
        self.assertEqual(result['reason'], 'pose_or_pairing_timeout')
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(result['attempts'][0]['diagnostics']['pose_outside_tolerance'], 7)
        self.assertEqual(active, {})

    async def test_semantic_rejection_is_not_retried_away(self):
        active = {}
        def health():
            active['result'] = {'view_id': 'a', 'status': 'rejected', 'reason': 'expected_target_absent'}
        result = await acquire_view({'view_id': 'a'}, move=AsyncMock(return_value='data: true'),
                                    active=active, get_fence=lambda: 1, check_fatal=health)
        self.assertEqual(result['reason'], 'expected_target_absent')
        self.assertEqual(len(result['attempts']), 1)

    async def test_cancellation_clears_active_attempt(self):
        active = {}
        def health():
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await acquire_view({'view_id': 'a'}, move=AsyncMock(return_value='data: true'),
                               active=active, get_fence=lambda: 1, check_fatal=health)
        self.assertEqual(active, {})

    def test_resume_checks_payload_and_preserves_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'rgb'; image.write_bytes(b'image')
            depth = root/'depth'; depth.write_bytes(b'depth')
            row = {'view_id': 'a', 'status': 'captured', 'split': 'development', 'map_id': 'simple',
                   'rgb_path': str(image), 'depth_path': str(depth),
                   'image_sha256': file_sha256(image), 'depth_sha256': file_sha256(depth)}
            receipt = root/'receipt.json'
            original = write_record(receipt, {'plan_identity': 'plan', 'mode': 'calibration',
                                             'map_id': 'simple', 'views': [row], 'status': 'blocked'})
            plan = {'identity': 'plan', 'map_id': 'simple'}
            rows, identity = resumed_views(receipt, plan, 'calibration', [{'view_id': 'a'}])
            self.assertEqual(identity, original['identity'])
            self.assertEqual(rows[0]['resumed_from_collection_identity'], identity)
            with self.assertRaises(ValueError):
                resumed_views(receipt, plan, 'pilot', [{'view_id': 'a'}])
            image.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                resumed_views(receipt, plan, 'calibration', [{'view_id': 'a'}])
