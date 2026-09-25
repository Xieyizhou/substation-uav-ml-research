"""A class-count feasibility witness, NOT a training protocol or sampler."""
from collections import Counter
from pathlib import Path
from scripts.vision.finalize_remaining_bridge_review import OUT,SOURCE,read,save,file_sha256,verify_tree

def main():
    path=OUT/'repair-budget-feasibility.json'
    if path.exists():verify_tree(path);return
    verify_tree(OUT/'diagnosis.json');r=read(OUT/'diagnosis.json')
    good=[(k,g) for k,g in sorted(r['groups'].items()) if not g['content_risk_ids']]
    # Explicit witness in sorted verified-lineage order; no seed or image selection.
    draws=[20,39,18,19,24]
    if len(good)!=len(draws):raise ValueError('Checked lineage scope changed')
    total=Counter();witness=[]
    for (key,group),n in zip(good,draws):
        vector=group['full_instances_per_variant']
        for cls,count in vector.items():total[cls]+=n*count
        witness.append(dict(lineage_id=key,draws_per_600_total_images=n,full_instance_vector=vector))
    prior=read(SOURCE.parent/'diagnosis.json');target=prior['constraints']['bridge_class_counts_per_600']
    if sum(draws)!=120 or dict(total)!=target:raise ValueError('Class-count witness failed')
    save(path,dict(status='arithmetic_feasibility_only_not_frozen_for_training',witness=witness,
        bridge_draws=120,full_instance_exposures=dict(total),independent_lineages=5,
        next_design_status='requires_new_protocol_and_matching_scale_lineage_controls',
        limitations=['Preserves only the four class-instance totals and bridge draw budget.',
            'Does not preserve old 13-lineage coverage, instance identities, scales or exposure positions.',
            'Does not include the eight new poses; the previous incompatibility proof for that combined design still applies.',
            'Not a concrete member schedule, not training admission, not evidence of model improvement.'],
        inputs={str(p):file_sha256(p) for p in (OUT/'diagnosis.json',SOURCE.parent/'diagnosis.json',Path(__file__))}))
    print('ARITHMETIC_WITNESS_VALID',dict(total),flush=True)

if __name__=='__main__':main()
