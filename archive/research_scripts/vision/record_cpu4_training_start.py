"""Snapshot evidence that a matched CPU4 experiment has reached real optimization."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import yaml
from src.ml.artifacts import file_sha256,object_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.contrast_cpu4_control import ROOT,KEYS,checked

def main():
    out=ROOT/'reference';p=checked(out/'protocol.json');ready=checked(out/'entry-ready.json');launch=checked(out/'launch-readiness.json')
    if ready['actual_draws_verified']!=8640 or launch['test_returncode']!=0 or launch['integrity']['pinned_files_verified']!=40:raise ValueError('Launch gate invalid')
    records=[];deps={}
    for key in KEYS:
        r=checked(out/'actual-preflight'/f'{key}.json')
        if r['actual']!=p['schedules'][key] or len(r['tensor_records'])!=480:raise ValueError('Incomplete preflight')
    for key in KEYS[:2]:
        attempt=out/'training'/key/'attempt-001/run';ap=attempt/'args.yaml';args=yaml.safe_load(ap.read_text())
        rows=list(csv.DictReader((attempt/'results.csv').open()))
        if not rows or int(rows[0]['epoch'])<1:raise ValueError('No completed optimizer epoch')
        if args['lr0']!=.00025 or args['batch']!=6 or args['nbs']!=6 or args['epochs']!=48 or args['optimizer']!='AdamW':raise ValueError('Configuration mismatch')
        # Every optimizer_step is fail-closed on get_num_threads()!=4 in the frozen worker.
        records.append(dict(key=key,first_epoch_snapshot=rows[0],snapshot_sha256=object_sha256(rows[0]),
            mutable_csv_source=str(attempt/'results.csv'),thread_evidence='Frozen optimizer-step guard rejects any actual count other than 4 before stepping; completed epoch proves guard traversal. Full ledger is written only at completion.'))
        deps[str(ap.resolve())]=file_sha256(ap)
    for f in (out/'protocol.json',out/'entry-ready.json',out/'launch-readiness.json',Path(__file__),ROOT/'plan-and-thread-erratum-zh.md'):deps[str(f.resolve())]=file_sha256(f)
    return write_record(ROOT/'training-start-receipt.json',dict(status='two_reference_seed_units_in_real_training_remaining_reference_and_contrast_automatically_queued',
        recorded_at=datetime.now(timezone.utc).isoformat(),records=records,numerical_evaluation_automatic=True,visual_error_review_pending=True,
        training_complete=False,model_passed=False,training_admitted=False,promotable=False,inputs=deps))

if __name__=='__main__':print(main()['status'])
