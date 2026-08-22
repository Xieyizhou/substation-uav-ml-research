"""Obstacle-aware free-space coverage from executed vehicle positions."""
import hashlib, json, math

def _line(a, b):
    x0,y0=a; x1,y1=b; dx,dy=abs(x1-x0),abs(y1-y0); sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1; error=dx-dy
    while True:
        yield x0,y0
        if (x0,y0)==(x1,y1): break
        twice=2*error
        if twice>-dy: error-=dy; x0+=sx
        if twice<dx: error+=dx; y0+=sy

class ExecutedCoverageTracker:
    def __init__(self, runtime_map, radius_m=2.0):
        self.map=runtime_map; self.radius_m=float(radius_m)
        self.free={(x,y) for x in range(runtime_map.width_cells) for y in range(runtime_map.height_cells) if (x,y) not in runtime_map.occupied_cells}
        self.covered=set(); self.samples=[]
        self.config_identity=hashlib.sha256(json.dumps({"map":runtime_map.map_identity,"radius_m":self.radius_m},sort_keys=True).encode()).hexdigest()
    def observe(self,east_m,north_m):
        origin=round(float(east_m)/self.map.resolution_m),round(float(north_m)/self.map.resolution_m); self.samples.append((round(float(east_m),6),round(float(north_m),6)))
        for cell in self.free:
            if math.dist(origin,cell)*self.map.resolution_m<=self.radius_m and not any(p in self.map.occupied_cells for p in list(_line(origin,cell))[1:-1]): self.covered.add(cell)
    @property
    def coverage(self): return len(self.covered)/len(self.free) if self.free else 1.0
    def receipt(self):
        trajectory=hashlib.sha256(json.dumps(self.samples,separators=(",",":")).encode()).hexdigest()
        return {"radius_m":self.radius_m,"covered_cell_count":len(self.covered),"free_cell_count":len(self.free),"coverage":round(self.coverage,8),"configuration_identity":self.config_identity,"trajectory_identity":trajectory}
