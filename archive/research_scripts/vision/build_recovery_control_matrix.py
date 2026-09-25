"""Freeze matched control candidates and diagnostic-only projected regions."""
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import quaternion, rotate, visible


def region(bounds, camera, orientation):
    inverse = [-v for v in orientation[:3]] + [orientation[3]]
    points = [rotate(inverse, [a-b for a,b in zip(p,camera)])
              for p in itertools.product(bounds[:2], bounds[2:4], bounds[4:6])]
    if any(p[0] <= 0 for p in points): return None
    focal = 960 / math.tan(1.466 / 2)
    pixels = [(960-focal*p[1]/p[0],540-focal*p[2]/p[0]) for p in points]
    return [min(p[0] for p in pixels),min(p[1] for p in pixels),
            max(p[0] for p in pixels),max(p[1] for p in pixels)]


def main():
    base = ROOT/'data/research/ml_training_recovery_v1'
    source = base/'paired-calibration-v1/plan-manifest.json'
    manifest = json.loads(source.read_text())
    inputs = {str(source):file_sha256(source), __file__:file_sha256(Path(__file__))}
    pairs = []
    for control in manifest['controls']:
        run = next(r for r in manifest['runs'] if r['source_plan_identity']==control['source_plan_identity'] and r['mode']=='full_2d')
        path = Path(run['plan_path']);plan=json.loads(path.read_text())
        obstacles = path.parent/'obstacles.json'
        for p in (path,obstacles):inputs[str(p)]=file_sha256(p)
        config=json.loads(obstacles.read_text());origin=config['gazebo_world_origin_m']
        names=sorted({v['object_id'] for v in control['views']})
        objects=[o for o in plan['objects'] if o['name'] in names]
        assert len(objects)==2
        for distance,height,bearing in itertools.product((6,10,16),(1.5,2.5,4),range(0,360,45)):
            views=[]
            for obj in objects:
                b=obj['bounds'];target=[(b[0]+b[1])/2,(b[2]+b[3])/2,(b[4]+b[5])/2]
                a=math.radians(bearing);camera=[target[0]+distance*math.cos(a),target[1]+distance*math.sin(a),height]
                q=quaternion(0,math.atan2(height-target[2],distance),math.atan2(target[1]-camera[1],target[0]-camera[0]))
                inside=origin[0]+.25<camera[0]<origin[0]+config['width']-.25 and origin[1]+.25<camera[1]<origin[1]+config['height']-.25
                occupied=any(o['bounds'][0]-.25<=camera[0]<=o['bounds'][1]+.25 and o['bounds'][2]-.25<=camera[1]<=o['bounds'][3]+.25 and o['bounds'][4]-.25<=height<=o['bounds'][5]+.25 for o in plan['objects'])
                # Nine interior rays describe configured occupancy, not pixel visibility.
                points=[target]+[[b[0]+u*(b[1]-b[0]),b[2]+v*(b[3]-b[2]),b[4]+w*(b[5]-b[4])] for u,v,w in itertools.product((.2,.8),repeat=3)]
                count=sum(visible(camera,p,plan['objects'],obj['name']) for p in points)
                roi=region(b,camera,q)
                clipped=roi is None or roi[0]<2 or roi[1]<2 or roi[2]>1918 or roi[3]>1078
                views.append({'object_id':obj['name'],'category':obj['category'],'bounds':b,'camera_position':camera,'orientation':q,
                              'diagnostic_projected_aabb':roi,'unobstructed_rays':count,'total_rays':9,
                              'occlusion_proxy':'clear' if count==9 else 'blocked' if count==0 else 'partial',
                              'exclusions':([ 'outside_world'] if not inside else [])+(['occupied_camera'] if occupied else [])+(['projected_bounds_clipped'] if clipped else [])})
            pair={'map_id':plan['map_id'],'source_plan_path':str(path),'distance':distance,'height':height,'bearing':bearing,'views':views,
                  'eligible_for_capture':all(not v['exclusions'] for v in views)}
            pair['identity']=object_sha256(pair);pairs.append(pair)
    eligible=[p for p in pairs if p['eligible_for_capture']]
    counts=Counter((p['map_id'], '/'.join(v['occlusion_proxy'] for v in p['views'])) for p in eligible)
    report={'schema_version':1,'inputs':inputs,'status':'matrix_prepared_not_captured','pairs':pairs,
            'candidate_pairs':len(pairs),'eligible_pairs':len(eligible),'eligible_views':len(eligible)*2,
            'strata':{':'.join(k):v for k,v in sorted(counts.items())},'training_admitted':False,
            'limits':['Projected occupancy AABB is a diagnostic region, not a visible mask or training label.',
                      'Nine-ray occupancy proxy must be verified against captured pixels; it is not measured occlusion.',
                      'Only camera poses are specified; vehicle extrinsics must be applied before collection.',
                      'No captures, predictions or threshold changes in this matrix preparation step.',
                      'Bounds, FOV and projection must be checked against sensor snapshot before region-based scoring.']}
    report['identity']=object_sha256(report)
    out=base/'control-matrix-v1';out.mkdir(exist_ok=True);write_json(out/'matrix.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','pairs')},indent=2))


if __name__=='__main__':main()
