"""Resolve SDF frames and conservatively bound simulation collision geometry."""
import json
import itertools
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from scripts.flight.fly_px4_shadow_hover import ROOT,PX4
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/vehicle-envelope-002'

def transform(pose):
    v=[0.]*6 if pose is None else list(map(float,pose.text.split()))
    if len(v)!=6 or not all(math.isfinite(x) for x in v):raise ValueError('Invalid pose')
    x,y,z,r,p,a=v;c,s=math.cos,math.sin
    rx=np.array([[1,0,0],[0,c(r),-s(r)],[0,s(r),c(r)]])
    ry=np.array([[c(p),0,s(p)],[0,1,0],[-s(p),0,c(p)]])
    rz=np.array([[c(a),-s(a),0],[s(a),c(a),0],[0,0,1]])
    t=np.eye(4);t[:3,:3]=rz@ry@rx;t[:3,3]=[x,y,z];return t

def analyze(xml):
    model=ET.fromstring(xml).find('model')
    nodes={n.get('name'):n for n in model if n.tag in ('frame','link')}
    if len(nodes)!=sum(n.tag in ('frame','link') for n in model):raise ValueError('Duplicate frame')
    cache={'__model__':np.eye(4)}
    def resolve(name,seen=()):
        if name in cache:return cache[name]
        if name in seen or name not in nodes:raise ValueError('Unresolved/cyclic frame '+name)
        pose=nodes[name].find('pose')
        parent=pose.get('relative_to','__model__') if pose is not None else '__model__'
        cache[name]=resolve(parent,seen+(name,))@transform(pose)
        return cache[name]
    moving=set()
    for j in model.findall('joint'):
        if j.get('type')=='revolute':
            if j.find('pose') is not None or j.findtext('parent')!='base_link':raise ValueError('Unsupported joint pivot')
            moving.add(j.findtext('child'))
        elif j.get('type')!='fixed':raise ValueError('Unsupported joint')
    bounds=[]
    for link in model.findall('link'):
        name=link.get('name');lt=resolve(name)
        for collision in link.findall('collision'):
            pose=collision.find('pose')
            if pose is not None and pose.get('relative_to'):raise ValueError('Unsupported collision reference')
            size=collision.findtext('geometry/box/size')
            if size is None:raise ValueError('Unsupported collision shape')
            sides=list(map(float,size.split()))
            if len(sides)!=3 or not all(math.isfinite(v) and v>0 for v in sides):raise ValueError('Invalid box')
            ct=transform(pose)
            # Triangle bound also covers arbitrary rotation of each rotor box
            # around its link origin, and arbitrary vehicle roll/pitch/yaw.
            radius=float(np.linalg.norm(lt[:3,3])+np.linalg.norm(ct[:3,3])+np.linalg.norm(sides)/2)
            if name not in moving:
                full=lt@ct
                radius=max(float(np.linalg.norm((full@np.array([sx*sides[0]/2,sy*sides[1]/2,sz*sides[2]/2,1]))[:3])) for sx,sy,sz in itertools.product((-1,1),repeat=3))
            bounds.append(dict(link=name,collision=collision.get('name'),rotating=name in moving,origin_sphere_radius_m=radius))
    if len(bounds)!=9 or len(moving)!=4:raise ValueError('Unexpected collision/rotor inventory')
    mounts={n:resolve(n)[:3,3].tolist() for n in ('base_link','research_lidar_link','research_camera_link')}
    return dict(collision_bounds=bounds,collision_sphere_radius_m=max(b['origin_sphere_radius_m'] for b in bounds),resolved_model_relative_translations_m=mounts,role='simulation collision shapes only; not visual mesh or real-aircraft certification')

def main():
    model=ROOT/'simulation/models/x500_research/model.sdf'
    paths=[model,PX4/'Tools/simulation/gz/models/x500/model.sdf',PX4/'Tools/simulation/gz/models/x500_base/model.sdf',Path(__file__).resolve()]
    result=subprocess.run(['/opt/homebrew/bin/gz','sdf','-p',str(model)],env={**os.environ,'SDF_PATH':f'{ROOT}/simulation/models:{PX4}/Tools/simulation/gz/models'},capture_output=True,text=True,check=True,timeout=15)
    analysis=analyze(result.stdout)
    OUT.mkdir(exist_ok=False)
    (OUT/'resolved.sdf').write_text(result.stdout);(OUT/'resolver.log').write_text(result.stderr)
    write_record(OUT/'envelope.json',dict(**analysis,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths+[OUT/'resolved.sdf']}))
    print(json.dumps(analysis,indent=2))

if __name__=='__main__':main()
