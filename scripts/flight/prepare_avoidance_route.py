"""Offline collision-derived route preparation. No vehicle connection."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.flight.fly_px4_shadow_hover import ROOT
from src.planner.astar_grid import astar,simplify_grid_path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/avoidance-plan-001'

def collision_boxes(path):
    """Conservative horizontal AABBs, rejecting unsupported transforms."""
    boxes=[]
    def visit(node,translation=(0.,0.,0.),name=''):
        if node.tag=='include':raise ValueError('Unresolved included collision model')
        pose=node.find('pose')
        if pose is not None:
            if pose.attrib:raise ValueError('Unresolved relative pose')
            v=list(map(float,pose.text.split()))
            if len(v)!=6 or not all(math.isfinite(x) for x in v) or any(abs(x)>1e-9 for x in v[3:]):raise ValueError('Unsupported collision transform')
            translation=tuple(a+b for a,b in zip(translation,v[:3]))
        if node.tag in ('model','link','collision'):name+='/'+node.get('name','')
        if node.tag=='collision':
            g=node.find('geometry');x,y,z=translation
            if g.find('plane') is not None:
                if g.findtext('plane/normal')!='0 0 1' or z!=0:raise ValueError('Unsupported plane')
                return
            if g.find('box') is not None:dx,dy,dz=map(float,g.findtext('box/size').split())
            elif g.find('cylinder') is not None:
                dx=dy=2*float(g.findtext('cylinder/radius'));dz=float(g.findtext('cylinder/length'))
            else:raise ValueError('Unsupported collision geometry')
            if not all(math.isfinite(v) and v>0 for v in (dx,dy,dz)):raise ValueError('Invalid collision dimensions')
            # World origin -10,-10 to map east/north. Keep all heights;
            # do not assume the 2 m flight can safely overfly low objects.
            boxes.append(dict(name=name,x0=x+10-dx/2,x1=x+10+dx/2,y0=y+10-dy/2,y1=y+10+dy/2,z0=z-dz/2,z1=z+dz/2))
            return
        for child in node:
            if child.tag in ('world','model','link','collision','include'):visit(child,translation,name)
    visit(ET.parse(path).getroot())
    return boxes

def inside(p,b,pad=0.):
    return b['x0']-pad<=p[0]<=b['x1']+pad and b['y0']-pad<=p[1]<=b['y1']+pad

def build(boxes,start=(1.5,1.5),goal=(6.5,1.5),resolution=.1,clearance=1.8):
    # Cells use lattice coordinates, not the production cell-centre adapter.
    size=201;blocked=set()
    for i in range(size):
        for j in range(size):
            p=(i*resolution,j*resolution)
            if not(.5<=p[0]<=19.5 and .5<=p[1]<=19.5) or any(inside(p,b,clearance+resolution/2) for b in boxes):blocked.add((i,j))
    index=lambda p:tuple(round(v/resolution) for v in p)
    path=astar(index(start),index(goal),blocked,size,size,False)
    points=[(i*resolution,j*resolution) for i,j in simplify_grid_path(path)]
    # Independently verify every straight segment; touching is not allowed.
    for a,b in zip(points,points[1:]):
        steps=math.ceil(math.dist(a,b)/.01)
        for k in range(steps+1):
            p=tuple(x+(y-x)*k/steps for x,y in zip(a,b))
            if any(inside(p,box,clearance) for box in boxes):raise ValueError('Segment violates clearance')
    return points

def main():
    world=ROOT/'simulation/worlds/substation_simple.sdf'
    boxes=collision_boxes(world)
    fixture=dict(name='planned_wall',x0=3.9,x1=4.1,y0=1.,y1=2.,z0=0.,z1=4.)
    rejection=None
    try:points=build(boxes+[fixture])
    except ValueError as exc:
        rejection=str(exc);points=[]
    length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
    # Feasibility only: hypothetical lower swept envelope 1.4 m above world
    # ground. It is NOT enabled until vehicle geometry/altitude is verified.
    alternative=build([b for b in boxes if b['z1']>=1.4]+[fixture])
    alternative_length=sum(math.dist(a,b) for a,b in zip(alternative,alternative[1:]))
    OUT.mkdir(exist_ok=False)
    write_record(OUT/'route-preflight.json',dict(status='route_geometry_blocked' if rejection else 'offline_geometry_passed_runtime_not_ready',rejection=rejection,simulation_only=True,collision_boxes=boxes,fixture=fixture,map_points=points,path_length_m=length if points else None,speed_command_m_s=.2,clearance_m=1.8,clearance_role='conservative planar research envelope, not certified aircraft separation',height_filtered_hypothesis=dict(status='not_authorized_for_flight',unverified_lower_swept_height_m=1.4,map_points=alternative,path_length_m=alternative_length,ideal_travel_time_lower_bound_s=alternative_length/.2,maximum_displacement_m=max(math.dist(alternative[0],p) for p in alternative)),existing_runtime_limits=dict(max_displacement_m=1.,post_hover_timeout_s=45.,vision_duration_s=90.),requires_runtime_envelope_extension=True,live_lidar_registration_verified=False,flight_authorized_by_this_receipt=False,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in (world,Path(__file__).resolve(),ROOT/'src/planner/astar_grid.py',ROOT/'simulation/models/x500_research/model.sdf')}))
    print('rejection',rejection,'alternative_length',alternative_length,'minimum_seconds',alternative_length/.2)

if __name__=='__main__':main()
