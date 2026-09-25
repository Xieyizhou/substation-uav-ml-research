"""Reproducible stage checks; never launches another replay attempt."""
import subprocess
import sys
from scripts.vision.instance_visibility_diagnosis import OUT,ROOT,prepare,read,save,file_sha256,Path,ET

MODULES=['test_instance_visibility_diagnosis','test_held_visibility_followup','test_edge_depth_support','test_edge_label_sources','test_switchgear_condition_review','test_canonical_gates','test_canonical_recovery']

def main():
    p=prepare();world_checks=[]
    for f in p['frames']:
        if f['status']!='source_verified':continue
        tree=ET.parse(f['source_world'])
        for e in f['events']:
            model=tree.find(f".//model[@name='{e['object_id']}']")
            if model is None:raise ValueError('Missing world instance')
            labels=[int(x.text) for x in model.findall('.//visual/plugin/label')]
            visuals=model.findall('.//visual')
            if len(labels)!=len(visuals) or set(labels)!={e['runtime_label']}:
                raise ValueError('World visual labels disagree with plan')
            world_checks.append(dict(review_id=e['review_id'],visuals=len(visuals),runtime_label=e['runtime_label'],status='verified'))
    command=[sys.executable,'-m','unittest',*[f'tests.{m}' for m in MODULES],'-q']
    r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    paths=[Path(__file__),OUT/'protocol.json',ROOT/'tools/gz_visibility_capture_cleanup_fixed.cc',OUT/'gz_visibility_capture_cleanup_fixed']
    paths += [ROOT/'tests'/f'{m}.py' for m in MODULES]
    paths += [ROOT/'scripts/vision'/n for n in ('run_instance_visibility_window.py','finalize_instance_visibility.py','record_instance_visibility_reviews.py')]
    save(OUT/'tests.json',dict(passed=r.returncode==0,command=command,output=r.stdout+r.stderr,world_visual_label_checks=world_checks,
        cleanup_fix=dict(status='compiled_not_runtime_revalidated',change='Transport node is declared after queues, mutex and callbacks, so subscriptions are destroyed before callback state.',
            limit='No fourth retry; corrected binary was not used to replace any failed attempt.'),
        known_global_failures='Historical global failures not rerun; no full-repository pass claimed.',
        inputs={str(x):file_sha256(x) for x in paths}))
    print(r.stdout+r.stderr)
    if r.returncode:raise SystemExit(r.returncode)

if __name__=='__main__':main()
