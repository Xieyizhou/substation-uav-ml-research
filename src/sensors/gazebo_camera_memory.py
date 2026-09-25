"""Independent RGB-memory adapter; only consumed frames persist lossless raw evidence."""
import asyncio
from dataclasses import dataclass
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
from src.sensors.gazebo_camera import GazeboCameraSource,CameraSourceEvent
from src.sensors.gazebo_image_codec import _message_bytes,_pixel_description,_packed_rows,_canonical_rgb_image
from src.sensors.gazebo_visual_transport import header_sequence,message_timestamp,GAZEBO_SIM_CLOCK
from src.sensors.types import CameraFrame,CAMERA_RAW_LAYOUT
from src.sensors.camera_decoded import decoded_content_sha256

@dataclass(frozen=True)
class MemoryEvent(CameraSourceEvent):
    decoded: object=None
    raw_rgb: bytes=b''

def parse_memory(message,topic,index,received):
    width=int(message.get('width',0));height=int(message.get('height',0))
    if width<=0 or height<=0:raise ValueError('Invalid dimensions')
    pixel=message.get('pixel_format_type',message.get('pixelFormatType'))
    fmt,channels=_pixel_description(pixel);step=int(message.get('step',width*channels))
    raw=_message_bytes(message.get('data'));packed=_packed_rows(raw,width,height,step,channels)
    rgb=np.array(_canonical_rgb_image(packed,width,height,fmt));rgb.setflags(write=False)
    payload=rgb.tobytes();sha=hashlib.sha256(payload).hexdigest();seq=header_sequence(message)
    frame=CameraFrame(frame_id=f'gazebo-rgb-{index:09d}',source_id=f'gazebo_rgb:{topic}',sequence_number=index if seq is None else seq,
        capture_timestamp=message_timestamp(message),capture_clock_domain=GAZEBO_SIM_CLOCK,receive_monotonic_timestamp=received,
        width=width,height=height,payload_format='raw',pixel_format='rgb8',payload_relative_path=f'frames/{sha}.raw',payload_sha256=sha,
        metadata=dict(runtime_topic=topic,source_pixel_format=fmt,source_row_stride_bytes=step,source_payload_sha256=hashlib.sha256(raw).hexdigest(),sequence_provenance='local_receive_ordinal' if seq is None else 'gazebo_header',evidence_policy='only_consumed_frames_saved_content_addressed'))
    decoded=SimpleNamespace(image=SimpleNamespace(array=rgb,decoded_content_sha256=decoded_content_sha256(rgb)))
    return MemoryEvent(index,frame=frame,decoded=decoded,raw_rgb=payload)

def persist_event(event,root):
    path=root/event.frame.payload_relative_path
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata=root/'metadata.json'
    if not metadata.exists():
        with metadata.open('x') as f:json.dump(dict(raw_layout_contract=CAMERA_RAW_LAYOUT),f)
    elif json.loads(metadata.read_text()).get('raw_layout_contract')!=CAMERA_RAW_LAYOUT:raise ValueError('Invalid raw layout')
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=event.frame.payload_sha256:raise ValueError('Evidence hash mismatch')
    else:
        with path.open('xb') as f:f.write(event.raw_rgb)

class GazeboMemoryCameraSource(GazeboCameraSource):
    async def _read_event(self,timeout_s):
        line=await asyncio.wait_for(self._process.stdout.readline(),timeout_s)
        if not line:raise RuntimeError('Gazebo RGB stream ended')
        self._receive_index+=1;index=self._receive_index;received=time.monotonic()
        def decode():
            try:return parse_memory(json.loads(line),self.topic,index,received)
            except (ValueError,TypeError,OSError) as e:return CameraSourceEvent(index,invalid_reason=f'{type(e).__name__}: {e}')
        return asyncio.create_task(asyncio.to_thread(decode))
