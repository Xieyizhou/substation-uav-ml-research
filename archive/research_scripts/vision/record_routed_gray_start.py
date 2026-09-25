"""Verify real first epochs, not merely a launch lock or queued state."""
from pathlib import Path
from scripts.vision import record_material_routed_start as receipt
from scripts.vision.routed_gray_transfer_control import OUT,KEYS,checked,freeze
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    freeze()
    receipt.OUT=OUT;receipt.KEYS=KEYS
    path=OUT/'training-start-receipt.json'
    r=checked(path) if path.exists() else receipt.run()
    r.pop('identity',None)
    r['inputs'][str(path.resolve())]=file_sha256(path)
    r['inputs'][str(Path(__file__).resolve())]=file_sha256(__file__)
    dest=OUT/'training-start-verified.json'
    return checked(dest) if dest.exists() else write_record(dest,r)

if __name__=='__main__':print(run()['status'])
