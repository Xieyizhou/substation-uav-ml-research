"""Exercise actual deferred JSON parser for every development frame."""
import base64
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.vision import verify_material_shadow_fix as verify
from scripts.vision import material_shadow as sidecar
from src.sensors.gazebo_camera_latest import RawCameraEvent

def main():
    original=sidecar.infer
    def infer(model,rgb):
        h,w,_=rgb.shape
        msg=dict(width=w,height=h,step=w*3,pixel_format_type='RGB_INT8',data=base64.b64encode(rgb.tobytes()).decode(),header={'stamp':{'sec':1,'nsec':0}})
        event=RawCameraEvent(1,json.dumps(msg).encode(),'/test',1.).materialize()
        np.testing.assert_array_equal(rgb,event.decoded.image.array)
        return original(model,event.decoded.image.array)
    with patch.object(verify,'OUT',Path('data/research/material-shadow-v1/latest-offline-v1').resolve()),patch.object(sidecar,'infer',infer):verify.main()

if __name__=='__main__':main()
