"""Defer JSON/base64/RGB work until a latest raw message is selected."""
import asyncio
from dataclasses import dataclass
import json
import time
from src.sensors.gazebo_camera import GazeboCameraSource
from src.sensors.gazebo_camera_memory import parse_memory

@dataclass(frozen=True)
class RawCameraEvent:
    receive_index: int
    line: bytes
    topic: str
    received: float
    valid: bool=True

    def materialize(self):
        return parse_memory(json.loads(self.line),self.topic,self.receive_index,self.received)

class GazeboLatestMemorySource(GazeboCameraSource):
    async def events(self,*,timeout_s):
        if self._process is None:raise RuntimeError('Source not started')
        while True:
            line=await asyncio.wait_for(self._process.stdout.readline(),timeout_s)
            if not line:raise RuntimeError('Gazebo stream ended')
            self._receive_index+=1
            yield RawCameraEvent(self._receive_index,line,self.topic,time.monotonic())
