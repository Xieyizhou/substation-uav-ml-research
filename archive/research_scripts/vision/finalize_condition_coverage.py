"""Bind descriptive coverage and proposed experiment; never starts collection."""
from pathlib import Path
from collections import Counter
import math,subprocess,sys,xml.etree.ElementTree as ET
from scripts.vision.build_condition_coverage import OUT,FIT,TRACE,prior,xml_value,digest
from scripts.vision.test_body_material_applicability import baseline_verify

def main():
    mp=OUT/'matrix.json';m=prior.read(mp);prior.verify(m)
    p=prior.read(FIT/'protocol.json');prior.verify(p);tp=TRACE/'member-source-trace.json';trace=prior.read(tp);prior.verify(trace)
    tr={r['member_id']:r for r in trace['rows']};common=set.intersection(*(set(v['draws']) for v in p['models'].values()))
    paths=[mp,FIT/'protocol.json',tp,Path(__file__).resolve()];lighting=[]
    for row in p['rows']:
        if row['member_id'] not in common or not row['class_instances']:continue
        path=Path(tr[row['member_id']]['source_image']).parents[2]/'plan/world.sdf'
        if not path.exists():lighting.append(dict(member_id=row['member_id'],status='world_missing'));continue
        paths.append(path);t=ET.parse(path);ambient=t.findtext('.//scene/ambient');sun=[l.findtext('diffuse') for l in t.findall('.//light')]
        lighting.append(dict(member_id=row['member_id'],world=str(path),world_sha256=prior.file_sha256(path),ambient=ambient,light_diffuse=sun,
            status='explicit_ambient' if ambient is not None else 'implicit_ambient_not_resolved'))
    nearest=[]
    for d in m['development']:
        if d['variant']!='original':continue
        a=d['actual_carrier_pose'];pairs=[]
        for t in m['training']:
            b=t['actual_carrier_pose'];dist=math.dist(a['position'],b['position']);qa,qb=a['orientation'],b['orientation']
            dot=abs(sum(x*y for x,y in zip(qa,qb)))/math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb));angle=math.degrees(2*math.acos(min(1,dot)))
            pairs.append(dict(member_id=t['member_id'],distance_m=dist,orientation_degrees=angle))
        nearest.append(dict(view_id=d['view_id'],nearest_carrier_position=min(pairs,key=lambda x:x['distance_m']),
            near_pose_match=any(x['distance_m']<=.05 and x['orientation_degrees']<=1 for x in pairs),
            scope='33 source-verified rows; carrier reference, not optical-center or complete-dataset independence test'))
    tests=['tests.test_condition_coverage','tests.test_structure_fit_integrity','tests.test_structure_fit','tests.test_lineage_training_fit']
    run=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if run.returncode:raise ValueError(run.stderr)
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Fixed baseline changed')
    paths.append(prior.ROOT/'docs/results/ml_condition_view_coverage_20260910.md')
    prior.frozen(OUT/'completion.json',dict(status='coverage_complete_lighting_control_proposed_not_ready',positive_lighting_census=lighting,
        nearest_pose_audit=nearest,baseline=b,regression=dict(output=run.stderr,whole_repository_tested=False),
        training_ready=False,collection_started=False,training_started=False,
        unresolved=['Historical nine scoped rows lack original instance mapping and annotation-mode field; posthoc reviews do not retroactively certify original capture.',
                    '24 common positive worlds have implicit ambient; effective renderer default not resolved.',
                    'Exact future member list, replay capability, label consistency and exposure feasibility not frozen.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('COVERAGE_COMPLETE_NOT_READY',len(lighting),len(nearest))

if __name__=='__main__':main()
