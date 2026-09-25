"""Verify actual batch identities and losses for the three fixed-sequence endpoints."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_fixed_sequence_diagnosis import OUT,KEYS,SEEDS,read,save,file_sha256,verify_tree
from scripts.vision.finalize_hard_negative_coverage_training import losses
from src.ml.artifacts import object_sha256

def main():
    verify_tree(OUT/'completion.json');p=read(OUT/'protocol.json');c=read(OUT/'completion.json')
    actuals=[];curves={};cells={}
    for key in KEYS:
        cell=read(OUT/key/'completion.json');actual=read(cell['exposure_path'])
        if actual['draws']!=p['schedules'][key] or actual['summary']!=p['exposures'][key] or cell['optimizer_steps']!=100:raise ValueError('Exposure mismatch')
        actuals.append(actual['draws']);cells[key]=cell
        curves[key]=losses(Path(cell['exposure_path']).parent/'results.csv')
    if not all(a==actuals[0] for a in actuals):raise ValueError('Actual cross-seed sequences differ')
    batches=[actuals[0][i:i+6] for i in range(0,600,6)]
    for key in KEYS:
        args=Path(cells[key]['exposure_path']).parent/'args.yaml'
        import yaml
        record=yaml.safe_load(args.read_text())
        if record['seed']!=int(key.split('-')[-1]):raise ValueError('Training seed mismatch')
        if key==KEYS[0]:reference=record
        elif any(record[k]!=v for k,v in reference.items() if k not in ('seed','name','project','save_dir')):
            raise ValueError('Non-seed training setting differs')
    fields=('train/box_loss','train/cls_loss','train/dfl_loss')
    loss_equal=all([[r[n] for n in fields] for r in curves[k]]==[[r[n] for n in fields] for r in curves[KEYS[0]]] for k in KEYS)
    suites=['tests.test_fixed_sequence_diagnosis','tests.test_negative_anchor','tests.test_hard_negative_coverage_evaluation','tests.test_hard_negative_coverage','tests.test_canonical_gates','tests.test_canonical_diagnostic_absence','tests.test_canonical_recovery','tests.test_canonical_shutdown','tests.test_exposure_diagnosis','tests.test_visual_bridge_training','tests.test_visual_bridge_supplement','tests.test_paired_visual_factors']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stderr)
    result=save(OUT/'audit.json',dict(status='passed',identical_actual_sequences=True,identical_actual_batches=True,
        common_sequence_sha256=object_sha256(actuals[0]),common_batch_sha256=object_sha256(batches),
        optimizer_steps=300,image_exposures=1800,identical_loss_curves=loss_equal,loss_curves=curves,
        tests_output=t.stdout+t.stderr,inputs={str(q):file_sha256(q) for q in [OUT/'completion.json',Path(__file__)]+[ROOT/(s.replace('.','/')+'.py') for s in suites]}))
    print('loss_equal',loss_equal,'parameters_equal',c['identical_parameters'],'predictions_equal',c['identical_predictions'])
    print(t.stdout+t.stderr)
    print('FPR',c['group']['no_target'])
    print('metrics',{v:{m:c['group'][v][m]['values'] for m in ('planned_instance_hit_rate','instance_recall')} for v in ('original','material','background','lighting')})
    print('failed',[x for x in c['candidate_gates']['checks'] if not x['passed']])

if __name__=='__main__':main()
