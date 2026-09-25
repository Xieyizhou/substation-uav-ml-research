"""Bounded acquisition retries and identity-checked continuation of captures."""

import asyncio
import re
import time

from src.ml.artifacts import file_sha256
from .plan import read_record
from .gates import target_checks, validate_view_pose


def acknowledged(response):
    return re.search(r"\bdata\s*:\s*true\b", response) is not None


def resumed_views(receipt_path, plan, mode, views, *, config=None,
                  check_version=None, instance_mapping=None):
    if receipt_path is None:
        return [], None
    receipt = read_record(receipt_path)
    if receipt['plan_identity'] != plan['identity'] or receipt['mode'] != mode:
        raise ValueError('Resume receipt plan/mode mismatch')
    if receipt.get('map_id') != plan['map_id']:
        raise ValueError('Resume receipt map mismatch')
    allowed = {view['view_id'] for view in views}
    kept = []
    seen = set()
    for row in receipt['views']:
        if row['view_id'] in seen or row['view_id'] not in allowed:
            raise ValueError('Resume contains duplicate or unknown views')
        seen.add(row['view_id'])
        if row['status'] != 'captured':
            continue
        if check_version is not None and row.get('check_version') != check_version:
            raise ValueError('Resume view predates current collection checks')
        if row.get('split') != 'development' or row.get('map_id') != plan['map_id']:
            raise ValueError('Resume view split/map mismatch')
        for field, digest in [('rgb_path', 'image_sha256'), ('depth_path', 'depth_sha256')]:
            if file_sha256(row[field]) != row[digest]:
                raise ValueError('Resume captured payload changed')
        view = next(item for item in views if item['view_id'] == row['view_id'])
        if config is not None:
            validate_view_pose(view, config, actual_carrier=row['actual_pose']['position'])
        if instance_mapping is not None:
            checks = target_checks(view, row['raw_truth'], instance_mapping)
            if view.get('category') in {'transformer','switchgear','capacitor_bank','reactor'} and checks['planned_instance_present'] is not True:
                raise ValueError('Resume planned target instance is absent')
            if row.get('target_checks') != checks:
                raise ValueError('Resume target checks are missing or stale')
        kept.append({**row, 'resumed_from_collection_identity': receipt['identity']})
    return kept, receipt['identity']


async def acquire_view(view, *, move, active, get_fence, check_fatal,
                       timeout_s=10, max_attempts=3):
    """Require service ACK plus fresh stable evidence on every attempt."""
    attempts = []
    try:
        for number in range(1, max_attempts + 1):
            active.clear()
            fence = get_fence()
            started = time.monotonic()
            try:
                response = await move()
                if not acknowledged(response):
                    raise RuntimeError('pose_service_not_acknowledged')
            except (RuntimeError, asyncio.TimeoutError) as error:
                attempts.append({'attempt': number, 'reason': 'pose_service_not_acknowledged',
                                 'detail': str(error), 'elapsed_s': time.monotonic() - started})
                check_fatal()
                continue
            # The attempt token prevents a metadata waiter from writing into a later view.
            active.update(view=view, fence=fence, stable=0, result=None,
                          token=object(), diagnostics={})
            deadline = time.monotonic() + timeout_s
            while active['result'] is None and time.monotonic() < deadline:
                check_fatal()
                await asyncio.sleep(.02)
            row = active['result']
            attempts.append({'attempt': number, 'reason': row.get('reason', row['status']) if row else 'pose_or_pairing_timeout',
                             'diagnostics': dict(active['diagnostics']),
                             'elapsed_s': time.monotonic() - started})
            if row is not None:
                return {**row, 'attempts': attempts}
        return {'view_id': view['view_id'], 'status': 'rejected',
                'reason': attempts[-1]['reason'], 'attempts': attempts,
                'training_admitted': False}
    finally:
        active.clear()
