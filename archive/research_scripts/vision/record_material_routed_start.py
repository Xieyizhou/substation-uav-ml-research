"""Snapshot first completed optimization epochs, not a model completion claim."""
import csv
from datetime import datetime,timezone
from pathlib import Path
import yaml
from scripts.vision.material_routed_contrast_control import OUT,KEYS,checked
from src.ml.artifacts import file_sha256,object_sha256
from src.vision.canonical.plan import write_record

def run():
    p=checked(OUT/'protocol.json');ready=checked(OUT/'entry-ready.json');launch=checked(OUT/'launch-readiness.json')
    if ready['actual_draws_verified']!=8640 or launch['test_returncode']!=0 or not launch['integrity']['integrity_passed'] or launch['integrity']['pinned_files_verified']!=40:raise ValueError('Launch gate incomplete')
    deps={};rows=[]
    for key in KEYS:
        path=OUT/'actual-preflight'/f'{key}.json';r=checked(path)
        if r['actual']!=p['schedules'][key] or len(r['tensor_records'])!=480:raise ValueError('Incomplete real loader preflight')
        deps[str(path.resolve())]=file_sha256(path)
    for key in KEYS[:2]:
        folder=OUT/'training'/key/'attempt-001/run';args_path=folder/'args.yaml';a=yaml.safe_load(args_path.read_text())
        epochs=list(csv.DictReader((folder/'results.csv').open()))
        if not epochs or int(epochs[0]['epoch'])<1:raise ValueError('No complete real optimization epoch')
        if a['lr0']!=.00025 or a['epochs']!=48 or a['batch']!=6 or a['nbs']!=6 or a['optimizer']!='AdamW' or a['model']!=p['initialization']['path']:raise ValueError('Actual training configuration mismatch')
        rows.append(dict(key=key,first_epoch=epochs[0],snapshot_sha256=object_sha256(epochs[0]),mutable_log_source=str(folder/'results.csv'),
            thread_evidence='Frozen CPU4 worker checks get_num_threads()==4 at every optimizer_step before executing; completed first epoch traversed this fail-closed guard. Full 480-step ledger pending completion.'))
        deps[str(args_path.resolve())]=file_sha256(args_path)
    for path in (OUT/'protocol.json',OUT/'entry-ready.json',OUT/'launch-readiness.json',OUT/'research-plan-zh.md',Path(__file__)):deps[str(path.resolve())]=file_sha256(path)
    return write_record(OUT/'training-start-receipt.json',dict(status='seed7_and_seed17_in_real_training_seed27_and_numerical_evaluation_automatically_queued',
        observed_at=datetime.now(timezone.utc).isoformat(),records=rows,training_complete=False,model_passed=False,visual_review_pending=True,
        training_admitted=False,promotable=False,inputs=deps))

if __name__=='__main__':print(run()['status'])
