"""Bind completed unarmed scan registration and collision-envelope evidence."""
import json
from scripts.flight.check_lidar_registration import OUT
from scripts.flight.check_vehicle_envelope import OUT as ENVELOPE
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def main():
    paths=[OUT/'registration.json',OUT/'runtime/receipt.json',ENVELOPE/'envelope.json']
    for p in paths:
        value=json.loads(p.read_text())
        for name,digest in value.get('inputs',{}).items():
            if file_sha256(name)!=digest:raise ValueError('Changed evidence: '+name)
    registration=json.loads(paths[0].read_text());runtime=json.loads(paths[1].read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['inputs'].items():
        if file_sha256(name)!=digest:raise ValueError('Changed protocol input: '+name)
    passed=registration['passed'] and runtime['status']=='unarmed_health_check_passed' and runtime['final_armed'] is False and not runtime['flight_requested'] and runtime['owned_processes_exited']
    write_record(OUT/'completion.json',dict(status='static_lidar_registration_verified' if passed else 'blocked',dynamic_registration_verified=False,simulation_only=True,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths+[OUT/'protocol.json']}))
    print('static_registration_complete',passed)

if __name__=='__main__':main()
