"""Current fixed-BN error evidence; no automatic review decisions."""
from pathlib import Path
from unittest.mock import patch
from scripts.vision import routed_fixed_bn_control as arm
from scripts.vision import inspect_routed_scale_results as renderer
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=arm.OUT/'audit-v1'

def run():
    for k in arm.KEYS:arm.verified_unit(k)
    with patch.object(renderer,'arm',arm),patch.object(renderer,'OUT',OUT):
        renderer.run()
    paths=[OUT/'evidence-build-receipt.json',Path(__file__),Path(renderer.__file__)]
    dest=OUT/'current-arm-evidence-receipt.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='fixed_bn_current_errors_ready_review_pending',
        note='Renderer reuse only; legacy build receipt status text mentions scale. All input paths and current model identities are fixed-BN, not scale-arm observations.',
        training_admitted=False,promotable=False,
        inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
