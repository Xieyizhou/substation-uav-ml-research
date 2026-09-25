"""Close the bounded permutation cause investigation with verified intervention."""
import sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.validate_canonical_batch_fix import ROOT_CAUSE as OUT,OUT as FIX,KEYS,read,save,file_sha256,verify_tree

def main():
    paths=[OUT/'single-batch.json',OUT/'trace.json',OUT/'first-adam-update.json',FIX/'completion.json']
    for p in paths:verify_tree(p)
    c=read(FIX/'completion.json')
    if not all(c[k] for k in ('identical_parameters','identical_loss_curves','identical_predictions')):raise ValueError('Full intervention failed')
    protocol=read(FIX/'protocol.json')
    for k in KEYS:
        cell=read(FIX/k/'completion.json');actual=read(cell['exposure_path'])
        if cell['optimizer_steps']!=100 or actual['draws']!=protocol['schedules'][k]:raise ValueError('Actual full-run sequence changed')
    from scripts.vision.reference_batch_order import align_to_reference
    from scripts.vision.run_order_diagnosis import REFERENCE,OUT as ORDER
    f=read(REFERENCE/'protocol.json')['schedules']['F-100-7'];w=read(ORDER/'protocol.json')['schedules']['W-100-7']
    if align_to_reference(w,f)!=f:raise ValueError('Reference restoration failed')
    suites=['tests.test_reference_batch_order','tests.test_permutation_root_cause','tests.test_canonical_batch_order','tests.test_order_diagnosis','tests.test_fixed_sequence_diagnosis','tests.test_negative_anchor','tests.test_hard_negative_coverage_evaluation','tests.test_hard_negative_coverage','tests.test_canonical_gates','tests.test_canonical_diagnostic_absence','tests.test_canonical_recovery','tests.test_canonical_shutdown','tests.test_exposure_diagnosis','tests.test_visual_bridge_training','tests.test_visual_bridge_supplement','tests.test_paired_visual_factors']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    b=subprocess.run([sys.executable,'scripts/vision/verify_experiment_baseline.py'],capture_output=True,text=True)
    baseline=json.loads(b.stdout) if b.returncode==0 else {}
    diff=subprocess.run(['git','diff','--check'],capture_output=True,text=True)
    if t.returncode or diff.returncode or not baseline.get('integrity_passed') or baseline.get('pinned_files_verified')!=40:raise ValueError('Regression or integrity failed')
    paths += [Path(__file__),ROOT/'scripts/vision/reference_batch_order.py']+[ROOT/(s.replace('.','/')+'.py') for s in suites]
    r=save(OUT/'completion.json',dict(status='batch_permutation_cause_and_control_verified',scope='Permutation sensitivity only; not a claim to resolve every detection error.',
        cause='Permutation-dependent FP32 reductions and backward rounding perturb near-zero gradients; clipped gradients comparable to AdamW epsilon cause large relative updates, then assignment branches diverge.',
        verified_control='Canonical stable member order before dataset fetch and collation; requested and effective sequences separately recorded.',
        selected_candidate=None,tests_output=t.stdout+t.stderr,baseline=baseline,diff_check=diff.stdout+diff.stderr,
        inputs={str(p):file_sha256(p) for p in paths}))
    print(t.stdout+t.stderr);print('identity',r['identity'])
    result=read(FIX/f'evaluation-{KEYS[0]}.json')
    print('quality',result['negative_summary'],{v:{m:s[m] for m in ('planned_instance_hit_rate','instance_recall')} for v,s in result['summary'].items()})

if __name__=='__main__':main()
