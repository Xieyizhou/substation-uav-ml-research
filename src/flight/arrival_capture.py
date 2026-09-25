"""Bounded near-waypoint position capture, not reverse cruise permission."""
import math


class ArrivalCapture:
    def __init__(self):self.active=False

    def reset(self):self.active=False

    def update(self,distance):
        if not math.isfinite(distance) or distance<0:raise ValueError('Invalid target distance')
        if self.active and distance>.12:raise RuntimeError('Arrival capture drift exceeded')
        if distance<=.08:self.active=True
        return self.active
