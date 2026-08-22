"""Truth-blind synchronized RGB-D back-projection into local NED."""
from __future__ import annotations
import math

def _rotation(roll_deg, pitch_deg, yaw_deg):
    r, p, y = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy), (cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy), (-sp, sr*cp, cr*cp))

def _apply(matrix, vector):
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)

def localize_bbox(depth_m, bbox, intrinsics, vehicle_pose, camera_extrinsics=None):
    required = ("east_m", "north_m", "altitude_m", "roll_deg", "pitch_deg", "yaw_deg")
    if any(key not in vehicle_pose for key in required): return None
    fx, fy = float(intrinsics.get("fx", 0)), float(intrinsics.get("fy", 0))
    cx, cy, depth = float(intrinsics.get("cx", 0)), float(intrinsics.get("cy", 0)), float(depth_m)
    if fx <= 0 or fy <= 0 or not math.isfinite(depth) or not .2 <= depth <= 100: return None
    x1, y1, x2, y2 = map(float, bbox); u, v = (x1+x2)/2, (y1+y2)/2
    optical = ((u-cx)*depth/fx, (v-cy)*depth/fy, depth)
    body = (optical[2], optical[0], optical[1])
    ext = camera_extrinsics or {}
    body = _apply(_rotation(ext.get("roll_deg", 0), ext.get("pitch_deg", 0), ext.get("yaw_deg", 0)), body)
    body = (body[0] + float(ext.get("forward_m", 0)), body[1] + float(ext.get("right_m", 0)), body[2] + float(ext.get("down_m", 0)))
    ned = _apply(_rotation(vehicle_pose["roll_deg"], vehicle_pose["pitch_deg"], vehicle_pose["yaw_deg"]), body)
    return {"north_m": float(vehicle_pose["north_m"])+ned[0], "east_m": float(vehicle_pose["east_m"])+ned[1], "altitude_m": float(vehicle_pose["altitude_m"])-ned[2]}
