"""Fixed-scene geometry, telemetry guards and owned SITL process utilities.

Active equivalents of frozen experiment helpers. Historical sources stay intact;
current qualification binds this module independently.
"""
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import struct
import subprocess
import xml.etree.ElementTree as ET

from src.vision.canonical.plan import write_record
from src.planner.astar_grid import astar, simplify_grid_path

TYPES = dict(zip(['uint64_t','int64_t','uint32_t','int32_t','uint16_t',
                 'int16_t','uint8_t','int8_t','float','double','bool','char'],
                'QqIiHhBbfd?c'))

def read_samples(path, topics, allow_incomplete_tail=False):
    data = Path(path).read_bytes()
    if data[:7] != b'ULog\x01\x12\x35':
        raise ValueError('Not a ULog file')
    offset = 16
    formats, subscriptions = {}, {}
    while offset + 3 <= len(data):
        size, kind = struct.unpack_from('<HB', data, offset)
        body = data[offset+3:offset+3+size]
        if len(body) != size:
            if allow_incomplete_tail:return
            raise ValueError('Truncated ULog record')
        offset += 3 + size
        if kind == ord('F'):
            name, fields = body.decode().split(':', 1)
            formats[name] = fields
        elif kind == ord('A'):
            subscriptions[struct.unpack_from('<H', body, 1)[0]] = (body[0], body[3:].decode())
        elif kind == ord('D'):
            multi, name = subscriptions.get(struct.unpack_from('<H', body)[0], (0, ''))
            if name not in topics:
                continue
            pos, values = 2, {}
            for field in formats[name].split(';'):
                if not field:
                    continue
                dtype, key = field.split()
                # ULog omits trailing alignment padding in data records.
                if key.startswith('_padding'):
                    continue
                count = 1
                if '[' in dtype:
                    dtype, count = dtype[:-1].split('[')
                    count = int(count)
                fmt = '<' + str(count) + TYPES[dtype]
                value = struct.unpack_from(fmt, body, pos)
                pos += struct.calcsize(fmt)
                values[key] = value[0] if count == 1 else value
            yield name, multi, values

def check_spawn(config,x=-8.5,y=-8.5,radius=1.):
    ox,oy,_=config['gazebo_world_origin_m'];resolution=config['resolution_m'];u=x-ox;v=y-oy
    if not(radius<=u<=config['width']*resolution-radius and radius<=v<=config['height']*resolution-radius):raise ValueError('Envelope outside map')
    for b in config['obstacles']:
        if (b['x_min']*resolution-radius<=u<=(b['x_max']+1)*resolution+radius and b['y_min']*resolution-radius<=v<=(b['y_max']+1)*resolution+radius):raise ValueError('Envelope intersects '+b['name'])

def guard(state,now,origin,max_height_m=1.8,max_drift_m=1.):
    for name in ('local','attitude'):
        if name not in state or now-state[name][0]>1.:raise RuntimeError('Stale '+name)
    p=state['local'][1];a=state['attitude'][1]
    if not all(math.isfinite(v) for v in [now,*origin.values(),*p.values(),*a.values()]):raise RuntimeError('Nonfinite telemetry')
    height=origin['down']-p['down'];drift=math.hypot(p['north']-origin['north'],p['east']-origin['east'])
    if height>max_height_m or height<-.4 or drift>max_drift_m:raise RuntimeError('Position envelope exceeded')
    if abs(a['roll'])>20 or abs(a['pitch'])>20:raise RuntimeError('Attitude envelope exceeded')
    return height,drift

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

def segment_distance(point,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1];length2=dx*dx+dy*dy
    if length2==0:return math.dist(point,a)
    t=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2))
    return math.dist(point,(a[0]+t*dx,a[1]+t*dy))

def angle_error(a,b):return (a-b+180)%360-180

def cruise_guard(state,origin,frame,a,b,now):
    height,_=guard(state,now,origin,max_height_m=2.8,max_drift_m=6.)
    if not 1.75<=height<=2.25:raise RuntimeError('Cruise altitude excursion')
    p=state['local'][1];att=state['attitude'][1];point=frame.to_map(p['east'],p['north'])
    if max(abs(att['roll']),abs(att['pitch']))>5:raise RuntimeError('Cruise tilt excursion')
    distance=segment_distance(point,a,b)
    if distance>.18:raise RuntimeError('Route corridor excursion')
    if math.hypot(p['vn'],p['ve'])>.35:raise RuntimeError('Actual speed excursion')
    return point,distance

def evaluate(estimated,truth):
    if any(not all(math.isfinite(r[k]) for k in ('timestamp','x','y','heading')) for r in estimated+truth):
        raise ValueError('Nonfinite alignment input')
    pairs=[];used=set()
    for a in estimated:
        choices=[(abs(a['timestamp']-b['timestamp']),i,b) for i,b in enumerate(truth) if i not in used]
        if not choices:break
        dt,i,b=min(choices,key=lambda v:v[0])
        if dt>33334:continue
        used.add(i)
        yaw=math.degrees((a['heading']-b['heading']+math.pi)%(2*math.pi)-math.pi)
        pairs.append(dict(timestamp_us=a['timestamp'],skew_us=dt,east_offset_m=b['y']+10-a['y'],north_offset_m=b['x']+10-a['x'],heading_error_deg=yaw))
    if len(pairs)<10:raise ValueError(f'Only {len(pairs)} synchronized pairs')
    east=statistics.median(p['east_offset_m'] for p in pairs);north=statistics.median(p['north_offset_m'] for p in pairs)
    residual=max(math.hypot(p['east_offset_m']-east,p['north_offset_m']-north) for p in pairs)
    heading=max(abs(p['heading_error_deg']) for p in pairs)
    stable_resets=all(all(k in r for r in estimated) and len({r[k] for r in estimated})==1 for k in ('xy_reset_counter','heading_reset_counter'))
    passed=residual<=.05 and heading<=1. and stable_resets and abs(east-1.5)<=.25 and abs(north-1.5)<=.25
    return dict(passed=passed,pairs=pairs,local_frame=dict(east_offset_m=east,north_offset_m=north,rotation_deg=0.),max_position_residual_m=residual,max_heading_error_deg=heading,reset_counters_stable=stable_resets,
        interpretation='Heading error is not absorbed into map rotation: ENU/NED axes are fixed by simulator. Static registration does not verify moving-flight alignment.')

def point_box_distance(point,box):
    return math.sqrt(sum(max(lo-v,0.,v-hi)**2 for v,lo,hi in zip(point,(box['x0'],box['y0'],box['z0']),(box['x1'],box['y1'],box['z1']))))

async def stop(p):
    if p is None:return
    if p.returncode is None:
        os.killpg(p.pid,signal.SIGINT)
        try:await asyncio.wait_for(p.wait(),8)
        except asyncio.TimeoutError:os.killpg(p.pid,signal.SIGKILL);await p.wait()

async def command(*args):
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 8)
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace")[:1000])
        return stdout.decode(errors="replace")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def check_port_available(kind, port):
    with socket.socket(socket.AF_INET, kind) as sock:
        # Match the gRPC listener's TCP behavior: TIME_WAIT is not a live owner.
        # SO_REUSEPORT is deliberately absent; an active listener still refuses.
        if kind == socket.SOCK_STREAM:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as error:
            raise RuntimeError(f'Runtime port {port} unavailable: {error}') from error


def preflight(build):
    conflict = subprocess.run(['pgrep', '-x', 'px4'], capture_output=True)
    if conflict.returncode == 0:
        raise RuntimeError('Existing PX4 instance; refusing to share it')
    if conflict.returncode != 1:
        raise RuntimeError('Cannot check existing PX4 processes')
    for kind, port in ((socket.SOCK_DGRAM, 14547), (socket.SOCK_DGRAM, 14587), (socket.SOCK_STREAM, 50177)):
        check_port_available(kind, port)
    return build / 'bin/px4'


async def alignment(out, *, airborne=False):
    if not airborne:
        await asyncio.sleep(10)
    logs = list((out / 'runtime/px4-work/log').rglob('*.ulg'))
    if len(logs) != 1:
        raise ValueError('Ambiguous live log')
    rows = list(read_samples(logs[0], {'vehicle_local_position', 'vehicle_local_position_groundtruth'}, allow_incomplete_tail=True))
    end = max(r['timestamp'] for _, _, r in rows)
    duration = 3000000 if airborne else 5000000
    a, b = ([r for n, m, r in rows if n == topic and m == 0 and r['timestamp'] >= end-duration]
            for topic in ('vehicle_local_position', 'vehicle_local_position_groundtruth'))
    result = evaluate(a, b)
    if airborne:
        result['heading_good_for_control_all'] = bool(a) and all(r.get('heading_good_for_control') is True for r in a)
        result['horizontal_flight_ready'] = result['passed'] and result['heading_good_for_control_all']
    write_record(out / ('inair-alignment.json' if airborne else 'prearm-alignment.json'), result)
    if not result['passed'] or (airborne and not result['horizontal_flight_ready']):
        raise RuntimeError('Live coordinate/heading alignment failed')


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
