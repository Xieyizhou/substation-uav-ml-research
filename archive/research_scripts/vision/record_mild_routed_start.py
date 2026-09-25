"""Independent receipt binding actual completed initial optimization epochs."""
from pathlib import Path
from scripts.vision import record_material_routed_start as base
from scripts.vision.mild_routed_contrast_control import OUT,KEYS,checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    base.OUT=OUT;base.KEYS=KEYS;source=OUT/'training-start-receipt.json'
    r=checked(source) if source.exists() else base.run();r.pop('identity',None)
    r['inputs'][str(source.resolve())]=file_sha256(source);r['inputs'][str(Path(__file__).resolve())]=file_sha256(__file__)
    dest=OUT/'training-start-verified.json'
    return checked(dest) if dest.exists() else write_record(dest,r)

if __name__=='__main__':print(run()['status'])
