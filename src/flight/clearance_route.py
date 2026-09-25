"""Prefer clearance within the unchanged hard 1.8 m exclusion envelope."""
import heapq
import math
from src.planner.astar_grid import simplify_grid_path, reconstruct_path
from scripts.flight.prepare_avoidance_route import inside
from src.flight.live_obstacle_map import LiveObstacleMap


def clearance_route(boxes,start,goal):
    resolution=.1
    def point(c):return (c[0]*resolution,c[1]*resolution)
    def gap(p,b):
        return math.hypot(max(b['x0']-1.8-p[0],0,p[0]-b['x1']-1.8),max(b['y0']-1.8-p[1],0,p[1]-b['y1']-1.8))
    # Same lattice and hard forbidden cells as the existing planner.
    costs={}
    for i in range(201):
        for j in range(201):
            p=point((i,j))
            if not all(.5<=v<=19.5 for v in p) or any(inside(p,b,1.85) for b in boxes):continue
            slack=min((gap(p,b) for b in boxes),default=.3)
            costs[i,j]=1.+8.*max(0.,(.3-slack)/.3)
    a=tuple(round(v/resolution) for v in start);z=tuple(round(v/resolution) for v in goal)
    for name,c in [('Start',a),('Goal',z)]:
        if c not in costs:raise ValueError(f'{name} cell {c} is outside the map or blocked')
    queue=[(0.,a)];best={a:0.};parent={};done=set()
    while queue:
        _,c=heapq.heappop(queue)
        if c in done:continue
        if c==z:break
        done.add(c)
        for d in ((c[0]+1,c[1]),(c[0]-1,c[1]),(c[0],c[1]+1),(c[0],c[1]-1)):
            if d not in costs or d in done:continue
            score=best[c]+(costs[c]+costs[d])/2
            if score>=best.get(d,float('inf')):continue
            best[d]=score;parent[d]=c
            heapq.heappush(queue,(score+abs(d[0]-z[0])+abs(d[1]-z[1]),d))
    else:raise ValueError(f'No A* path found from {a} to {z}')
    cells=reconstruct_path(parent,z)
    # Shortcut staircase turns only when the sampled segment keeps both the
    # hard envelope and the available margin (capped at 0.15m) of that subpath.
    lattice=[point(c) for c in cells]
    gaps=[min((gap(p,b) for b in boxes),default=.3) for p in lattice]
    pruned=[lattice[0]];i=0
    while i<len(lattice)-1:
        for j in range(len(lattice)-1,i,-1):
            margin=min(.15,min(gaps[i:j+1]));a,b=lattice[i],lattice[j]
            steps=max(1,math.ceil(math.dist(a,b)/.01));safe=True
            for k in range(steps+1):
                p=tuple(x+(y-x)*k/steps for x,y in zip(a,b))
                if any(inside(p,box,1.8) or gap(p,box)<margin-1e-9 for box in boxes):
                    safe=False;break
            if safe:break
        if not safe:raise ValueError('No safe simplification segment')
        pruned.append(lattice[j]);i=j
    lattice=pruned
    # Keep the checked connector rather than cutting the first lattice turn.
    route=[tuple(start)]+lattice
    route=[p for i,p in enumerate(route) if i==0 or math.dist(p,route[i-1])>1e-9]
    for a,b in zip(route,route[1:]):
        n=max(1,math.ceil(math.dist(a,b)/.01))
        for i in range(n+1):
            p=tuple(x+(y-x)*i/n for x,y in zip(a,b))
            if any(inside(p,box,1.8) for box in boxes):raise ValueError('Segment violates clearance')
    return route


class ClearanceMap(LiveObstacleMap):
    def replan(self,start,goal):
        if not self.points:raise ValueError('No sensor obstacle evidence')
        return clearance_route(self.static_boxes+self.boxes(),start,goal)
