"""All 96 frames must preserve canonical pixels and historical predictions."""
import base64
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.vision import verify_material_shadow_fix as verify
from scripts.vision import material_shadow as sidecar
from src.sensors.gazebo_camera_memory import parse_memory

OUT=Path('data/research/material-shadow-v1/memory-offline-v1').resolve()

def main():
    original=sidecar.infer
    def checked(model,rgb):
        h,w,_=rgb.shape
        msg=dict(width=w,height=h,step=w*3,pixel_format_type='RGB_INT8',data=base64.b64encode(rgb.tobytes()).decode(),header={'stamp':{'sec':1,'nsec':0}})
        event=parse_memory(msg,'/test',1,1.)
        np.testing.assert_array_equal(rgb,event.decoded.image.array)
        return original(model,event.decoded.image.array)
    with patch.object(verify,'OUT',OUT),patch.object(sidecar,'infer',checked):verify.main()

if __name__=='__main__':main()
