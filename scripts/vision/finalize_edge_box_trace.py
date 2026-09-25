"""Bind stage-localization report; keep data and training blockers explicit."""
from pathlib import Path
from scripts.vision.trace_closed_material_edge_box import OUT,ROOT,read,verify,frozen,file_sha256


def main():
    dp=OUT/'diagnosis.json';r=read(dp);verify(r)
    dest=OUT/'completion.json'
    if dest.exists():verify(read(dest));print('VALID_TRACE_COMPLETION_REUSED');return
    if r['status']!='missing_instance_localized_upstream_of_python_conversion':raise ValueError('Invalid trace status')
    report=ROOT/'docs/results/ml_closed_material_edge_box_trace_20260910.md'
    frozen(dest,dict(status='stage_trace_complete_upstream_branch_unresolved',diagnosis_identity=r['identity'],
        training_ready=False,training_started=False,historical_labels_changed=False,renderer_modified=False,
        established='Instance 128 absent in source raw message and all 36 directly recorded replay box messages; current converter retains synthetic positive-area edge boxes.',
        unresolved=['Exact Gazebo runtime filtering branch','Explicit edge-fragment supervision/ignore policy','Independent T036 instance 118 content risk'],
        next_step='Isolated same-pose full_2d versus visible_2d diagnostic, not a production mode switch or label repair.',
        inputs={str(p):file_sha256(p) for p in (dp,report,Path(__file__))}))
    print('TRACE_COMPLETE; NO_TRAINING_OR_LABEL_CHANGES')


if __name__=='__main__':main()
