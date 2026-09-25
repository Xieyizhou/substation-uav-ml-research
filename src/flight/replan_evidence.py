"""Persist raw scan/pose pairs and maps, including terminal failures."""
import json
import math
from src.vision.canonical.plan import write_record


def scan_record(scan):
    if scan is None:return None
    r=scan.to_record()
    # JSON has no infinity value; float('inf') restores the exact range state.
    r['ranges_m']=[v if math.isfinite(v) else str(v) for v in r['ranges_m']]
    return r


class EvidenceJournal:
    def __init__(self,out,frame):
        self.out=out;self.frame=vars(frame);self.last_sequence=None;self.count=0
        self.handle=(out/'scan-pose.jsonl').open('x')

    def append(self,scan,state,now,yaw_rate,eligible,added):
        self.last_sequence=scan.sequence
        record=dict(scan=scan_record(scan),local=state['local'],attitude=state['attitude'],now=now,yaw_rate=yaw_rate,map_eligible=eligible,new_returns=added,local_frame=self.frame)
        self.handle.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');self.handle.flush()

    def snapshot(self,reason,mapping,state,scan,route,index):
        self.count+=1
        path=self.out/f'event-map-{self.count:03}.json'
        write_record(path,dict(reason=reason,source='live_lidar_only',points=list(mapping.points),boxes=mapping.boxes(),local=state['local'],attitude=state['attitude'],scan=scan_record(scan),local_frame=self.frame,route=route,index=index))
        return path.name

    def close(self):self.handle.close()
