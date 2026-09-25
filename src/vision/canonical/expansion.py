"""Deterministic bounded expansion for canonical target and no-target views."""

import math

from src.vision.canonical.occlusion import view_blockers
from src.vision.canonical.plan import object_sha256, quaternion, rotate


CLASSES = ("transformer", "switchgear", "capacitor_bank", "reactor")


def _center(bounds):
    return tuple((bounds[i] + bounds[i + 1]) / 2 for i in (0, 2, 4))


def _pose(camera, look_at, yaw_offset):
    dx, dy, dz = (look_at[i] - camera[i] for i in range(3))
    horizontal = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx) + math.radians(yaw_offset)
    pitch = math.atan2(-dz, horizontal)
    q = quaternion(0.0, pitch, yaw)
    offset = rotate(q, (0.18, 0.0, 0.12))
    return [camera[i] - offset[i] for i in range(3)], list(q)


def target_candidates(map_id, objects, origin, size, round_index):
    """Generate continuous target views without changing canonical geometry."""
    xmin, ymin = origin
    xmax, ymax = xmin + size[0], ymin + size[1]
    rows = []
    for obj in objects:
        if obj['category'] not in CLASSES:
            continue
        target = _center(obj['bounds'])
        for step in range(72):
            bearing = (step * 5 + round_index * 2) % 360
            distance = 5.0 + ((step * 7 + round_index * 3) % 27) * 0.5
            height = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)[(step + round_index) % 6]
            angle = math.radians(bearing)
            camera = [target[0] + distance * math.cos(angle),
                      target[1] + distance * math.sin(angle), height]
            if not (xmin + .25 <= camera[0] <= xmax - .25 and
                    ymin + .25 <= camera[1] <= ymax - .25):
                continue
            offset = (-12, -6, 0, 6, 12)[(step * 3 + round_index) % 5]
            position, orientation = _pose(camera, target, offset)
            row = {'map_id': map_id, 'object_id': obj['name'],
                   'category': obj['category'], 'bearing': bearing,
                   'distance': distance, 'height': height, 'offset': offset,
                   'camera_position': camera, 'position': position,
                   'orientation': orientation,
                   'family': f"{map_id}:{obj['name']}:{bearing // 15}"}
            row['view_id'] = object_sha256(row)
            if not view_blockers(row, objects):
                rows.append(row)
    return rows


def background_candidates(map_id, objects, origin, size, round_index):
    """Aim inward from boundary-adjacent cameras at empty ground strips."""
    xmin, ymin = origin
    xmax, ymax = xmin + size[0], ymin + size[1]
    rows = []
    for step in range(240):
        side = step % 4
        t = ((step * 37 + round_index * 11) % 997) / 997
        if side == 0:
            camera, look = [xmin + .5, ymin + .5 + t*(size[1]-1), 2.0], [xmin + 2.5, ymin + .5 + t*(size[1]-1), .2]
        elif side == 1:
            camera, look = [xmax - .5, ymin + .5 + t*(size[1]-1), 2.5], [xmax - 2.5, ymin + .5 + t*(size[1]-1), .2]
        elif side == 2:
            camera, look = [xmin + .5 + t*(size[0]-1), ymin + .5, 3.0], [xmin + .5 + t*(size[0]-1), ymin + 2.5, .2]
        else:
            camera, look = [xmin + .5 + t*(size[0]-1), ymax - .5, 3.5], [xmin + .5 + t*(size[0]-1), ymax - 2.5, .2]
        position, orientation = _pose(camera, look, (-10, 0, 10)[step % 3])
        row = {'map_id': map_id, 'object_id': 'empty_ground',
               'category': 'no_target', 'bearing': step, 'distance': 2.0,
               'height': camera[2], 'offset': (-10, 0, 10)[step % 3],
               'camera_position': camera, 'position': position,
               'orientation': orientation,
               'family': f"{map_id}:no_target:{side}:{step // 12}"}
        row['view_id'] = object_sha256(row)
        rows.append(row)
    return rows


def balanced_select(targets, backgrounds, seen, limit=120):
    groups = {name: [] for name in (*CLASSES, 'no_target')}
    for row in (*targets, *backgrounds):
        if row['view_id'] not in seen:
            groups[row['category']].append(row)
    for values in groups.values():
        values.sort(key=lambda row: row['view_id'])
    selected = []
    while len(selected) < limit and any(groups.values()):
        for name in (*CLASSES, 'no_target'):
            if groups[name]:
                selected.append(groups[name].pop(0))
                if len(selected) == limit:
                    break
    return selected
