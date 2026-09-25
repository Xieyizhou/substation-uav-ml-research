"""Render current late-decay evidence without creating review decisions."""
from unittest.mock import patch
from scripts.vision import routed_late_decay_control as arm
from scripts.vision import inspect_routed_scale_results as renderer

def run():
    for k in arm.KEYS:arm.verified_unit(k)
    with patch.object(renderer,'arm',arm),patch.object(renderer,'OUT',arm.OUT/'audit-v1'):
        return renderer.run()

if __name__=='__main__':print(run()['status'])
