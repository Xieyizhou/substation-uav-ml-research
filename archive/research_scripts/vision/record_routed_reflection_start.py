"""Audit the goal endpoint using completed optimization epochs and frozen inputs."""
from pathlib import Path
from scripts.vision import record_material_routed_start as base
from scripts.vision.routed_reflection_control import OUT,KEYS,checked,freeze
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    freeze();base.OUT=OUT;base.KEYS=KEYS
    path=OUT/'training-start-receipt.json'
    r=checked(path) if path.exists() else base.run();r.pop('identity',None)
    paths=[path,Path(__file__),OUT/'execution-identity.json',OUT/'reflection-exposure-ledger.json']
    for p in paths:
        if p.suffix=='.json':checked(p)
        r['inputs'][str(p.resolve())]=file_sha256(p)
    return write_record(OUT/'training-start-verified.json',r)

if __name__=='__main__':print(run()['status'])
