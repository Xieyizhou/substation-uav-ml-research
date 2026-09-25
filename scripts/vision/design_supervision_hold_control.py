"""Design-only impact ledger. No dataset export or training entry point."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.vision.structure_fit import OUT as FIT, SOURCE, read, verify, frozen, file_sha256, ROOT
from scripts.vision.resume_supervision_risk_pilot import OUT as PILOT

OUT = SOURCE / 'whole-image-hold-control-design-v1'
EVENTS = ('T08', 'T29', 'T30', 'T33')

def ledger(rows, draws, held):
    index = {r['member_id']: r for r in rows}
    if len(index) != len(rows) or set(draws) - set(index) or not set(held) <= set(index):
        raise ValueError('Duplicate or unresolved member')
    if len(draws) != 2700:
        raise ValueError('Expected 450 complete batches')
    removed = Counter()
    for mid in held:
        removed.update(index[mid]['class_instances'])
    slots = [dict(exposure_index=i, step=i//6+1, batch_offset=i%6,
                  member_id=m, subset=index[m]['subset'])
             for i, m in enumerate(draws) if m in held]
    return dict(held_full_labels=dict(removed), affected_slots=slots,
                affected_exposures=len(slots), affected_batches=len({s['step'] for s in slots}),
                affected_subset_exposures=dict(Counter(s['subset'] for s in slots)))

def build():
    paths = [FIT/'protocol.json', PILOT/'completion.json', PILOT/'updated-proposal.json']
    p, completion, proposal = [read(x) for x in paths]
    for doc in (p, completion, proposal):
        verify(doc)
    if completion['status'] != 'revision_proposal_ready_not_applied':
        raise ValueError('Prior proposal incomplete')
    decisions = proposal['decisions']
    if len({d['event_id'] for d in decisions}) != len(decisions):
        raise ValueError('Duplicate proposal decision')
    if sorted(d['event_id'] for d in decisions if d['proposal']=='hold_whole_image_proposed') != list(EVENTS):
        raise ValueError('Unexpected hold scope')
    events = [e for e in p['reactors'] if e['event_id'] in EVENTS]
    held = {e['member']['member_id'] for e in events}
    if len(events) != 4 or len(held) != 4:
        raise ValueError('Ambiguous event-to-image identity')
    if len(p['rows']) != 236:
        raise ValueError('Pool changed')
    models = {str(seed): ledger(p['rows'], p['models'][f'interleaved-450-{seed}']['draws'], held)
              for seed in (7,17,27)}
    paths += [Path(__file__), ROOT/'docs/whole-image-hold-control-plan-v1.md']
    return frozen(OUT/'design.json', dict(status='design_ready_execution_not_authorized',
        held_events=list(EVENTS), held_member_ids=sorted(held), current_members=236,
        hypothetical_retained_members=232, seed_impact=models,
        proposed_arms=['unchanged_interleaved', 'same_subset_slot_replacement'],
        interpretation='Whole-image withholding plus same-subset replacement policy, not isolated label quality causality.',
        blockers_before_training=['explicit_execution_authorization', 'registered_lineage_closure_recheck',
          'replacement_eligibility_and_source_review', 'exact_replacement_sequence_and_full_instance_ledger',
          'real_loader_preflight_all_six_units', 'fixed_evaluation_and_retention_gate_binding'],
        labels_modified=False, dataset_exported=False, sampling_generated=False, training_started=False,
        inputs={str(x):file_sha256(x) for x in paths}))

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--design',action='store_true');args=ap.parse_args()
    if args.design:
        OUT.mkdir(exist_ok=True);build();print('DESIGN_READY_NO_DATASET_NO_TRAINING')
    else:
        print('PREFLIGHT_ONLY_NO_DATASET_NO_TRAINING')
