"""Seed-keyed final verification and report material for the anchor experiment."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_negative_anchor import OUT,PREV,SEEDS,read,save,file_sha256,verify_tree,aggregate
from scripts.vision.finalize_hard_negative_coverage_training import losses

def main():
    verify_tree(OUT/'completion.json');c=read(OUT/'completion.json');p=read(OUT/'protocol.json');old=read(PREV/'completion.json')
    rows=[read(OUT/f'evaluation-H-100-{s}.json') for s in SEEDS]
    if c['group']!=aggregate(rows):raise ValueError('Seed order or aggregation mismatch')
    curves={}
    for s in SEEDS:
        cell=read(OUT/f'H-100-{s}/completion.json');e=read(cell['exposure_path'])
        if e['draws']!=p['schedules'][f'H-100-{s}'] or cell['optimizer_steps']!=100:raise ValueError('Actual exposure mismatch')
        curves[str(s)]=losses(Path(cell['exposure_path']).parent/'results.csv')
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_negative_anchor','tests.test_hard_negative_coverage','tests.test_exposure_diagnosis','tests.test_hard_negative_coverage_evaluation'],capture_output=True,text=True)
    if tests.returncode:raise ValueError(tests.stderr)
    paired=[]
    for s,row in zip(SEEDS,rows):
        for arm in ('O','N'):
            other=read(PREV/f'{arm}-100-{s}.json');lookup={(r['pair_id'],r['variant']):r for r in other['rows']}
            for r in row['rows']:
                previous=lookup[r['pair_id'],r['variant']]
                paired.append(dict(seed=s,reference=arm,pair_id=r['pair_id'],variant=r['variant'],category=r['category'],
                    planned_hit_delta=int(r['planned_instance_hit'])-int(previous['planned_instance_hit']),
                    matched_instance_delta=len(r['matches'])-len(previous['matches'])))
    save(OUT/'audit.json',dict(status='passed',seed_order=list(SEEDS),actual_optimizer_steps=300,image_exposures=1800,
        loss_curves=curves,loss_scope='Training-member fitting only',paired_deltas=paired,tests_output=tests.stdout+tests.stderr,
        inputs={str(q):file_sha256(q) for q in [OUT/'completion.json',Path(__file__),ROOT/'tests/test_negative_anchor.py',ROOT/'scripts/vision/resume_negative_anchor.py']}))
    print('sampling_hypothesis_supported',c['sampling_hypothesis_supported'],'candidate',c['selected_candidate'])
    for a,g in [('O',old['groups']['O']),('N',old['groups']['N']),('H',c['group'])]:
        print(a,'FPR',g['no_target']['frame_false_positive_rate']['values'])
        for v in ('original','material','background','lighting'):
            print(a,v,'planned',g[v]['planned_instance_hit_rate']['mean'],'recall',g[v]['instance_recall']['mean'])
    print('retention',c['relative_N_retention'])
    print('failed_gates',[r for r in c['candidate_gates']['checks'] if not r['passed']])

if __name__=='__main__':main()
