"""Verify the corrected final analysis without overwriting frozen evidence."""
import subprocess,sys
from pathlib import Path
from scripts.vision.correct_material_trajectory_labels import OUT,prior,classify

def main():
    paths=[OUT/'completion.json',OUT/'summary-v2.json',OUT/'transition-windows.json']
    records=[prior.read(p) for p in paths]
    for r in records:prior.verify(r)
    s=records[1]
    for t in s['trajectories']:
        if t['classification']!=classify([x['hit'] for x in t['timeline']]):raise ValueError('Classification conflict')
    original=prior.read(OUT/'summary.json')
    if s['counts']!=original['counts'] or s['development']!=original['development']:raise ValueError('Unexpected metric change')
    test=subprocess.run([sys.executable,'-m','unittest','tests.test_material_trajectory_label_correction'],capture_output=True,text=True,timeout=30)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    paths.extend([Path(__file__).resolve(),prior.ROOT/'tests/test_material_trajectory_label_correction.py',
                  prior.ROOT/'docs/results/ml_material_learning_trajectory_erratum_20260912.md'])
    return prior.frozen(OUT/'completion-v2.json',dict(status='trajectory_analysis_complete_with_classification_erratum',
        previous_complete_units=records[0]['inference_units_verified'],baseline=records[0]['baseline'],
        corrected_labels=len(s['erratum']['changes']),previous_regression_output=records[0]['regression_output'],
        correction_regression_output=test.stdout+test.stderr,selected_candidate=None,
        inputs={str(p):prior.file_sha256(p) for p in paths}))

if __name__=='__main__':print(main()['status'])
