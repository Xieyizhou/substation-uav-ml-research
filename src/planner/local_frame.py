"""Explicit horizontal map (east,north) <-> PX4 local NED registration.

Offsets locate the PX4 local origin in map metres. Rotation is mathematical
counterclockwise from local east to map east, NOT vehicle compass yaw.
This transform does not certify calibration or alter vertical datum.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LocalFrame:
    east_offset_m: float = 0.
    north_offset_m: float = 0.
    rotation_deg: float = 0.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.east_offset_m,self.north_offset_m,self.rotation_deg)):
            raise ValueError('Nonfinite local frame registration')

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):return value
        return cls(**(value or {}))

    def to_map(self, east, north):
        self._finite(east,north)
        a=math.radians(self.rotation_deg);c,s=math.cos(a),math.sin(a)
        return self.east_offset_m+c*east-s*north,self.north_offset_m+s*east+c*north

    def to_local(self, east, north):
        self._finite(east,north)
        e,n=east-self.east_offset_m,north-self.north_offset_m
        a=math.radians(self.rotation_deg);c,s=math.cos(a),math.sin(a)
        return c*e+s*n,-s*e+c*n

    def cell(self, east, north, resolution):
        if not math.isfinite(resolution) or resolution<=0:raise ValueError('Invalid resolution')
        e,n=self.to_map(east,north)
        return math.floor(e/resolution),math.floor(n/resolution)

    @staticmethod
    def _finite(*values):
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite position')
