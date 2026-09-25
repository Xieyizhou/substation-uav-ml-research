"""Observe live formal optimization, never confuse launch with training."""
import csv, io, subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from scripts.vision.physical_low_light_design import OUT, KEYS, prior
from scripts.vision.train_physical_low_light import ready

def main():
    import torch
    ready(); observations=[]
    ps=subprocess.check_output(['ps','-axo','pid,etime,%cpu,command'],text=True)
    for key in KEYS[:2]:
        processes=[line for line in ps.splitlines() if 'scripts.vision.train_physical_low_light --worker '+key+' --train' in line]
        if len(processes)!=1:raise ValueError('Worker not uniquely live')
        roots=list((OUT/'training'/key).glob('attempt-*/run'))
        if len(roots)!=1:raise ValueError('Ambiguous running attempt')
        root=roots[0];payload=(root/'weights/last.pt').read_bytes()
        c=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        optimizer=c.get('optimizer')
        if not optimizer:raise ValueError('No actual optimizer state')
        counts=[int(s['step']) for s in optimizer['state'].values() if 'step' in s]
        if not counts or min(counts)<1:raise ValueError('No optimization evidence')
        curve=(root/'results.csv').read_text();rows=list(csv.DictReader(io.StringIO(curve)))
        if not rows or int(c['epoch'])<0:raise ValueError('No completed epoch')
        observations.append(dict(cell=key,process=processes[0],checkpoint_snapshot_sha256=sha256(payload).hexdigest(),
            checkpoint_is_mutable_running_snapshot_not_endpoint=True,epoch_zero_based=int(c['epoch']),optimizer_parameter_steps_min=min(counts),optimizer_parameter_steps_max=max(counts),
            curve_snapshot=rows,curve_snapshot_sha256=sha256(curve.encode()).hexdigest()))
    return prior.frozen(OUT/'training-start-confirmed.json',dict(status='formal_training_started_not_finished',observed_at=datetime.now(timezone.utc).isoformat(),observations=observations,
        queued_cells=list(KEYS[2:]),cpu='two concurrent workers, four threads each; remaining four units follow automatically',evaluation_started=False,
        inputs={str(p):prior.file_sha256(p) for p in (OUT/'entry-ready.json',OUT/'training-authorization.json',Path(__file__).resolve())}))

if __name__=='__main__':print(main()['status'])
