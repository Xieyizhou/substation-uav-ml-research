"""Finalize semantic decisions for review-only background crops."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json


def main():
    base=ROOT/'data/research/ml_training_recovery_v1/background-hard-negative-candidates-v1'
    source=base/'manifest.json';manifest=json.loads(source.read_text())
    held={11}
    subtype={2:'pole_and_roof_edge',4:'edge_only',5:'edge_only',6:'edge_only',7:'roof_edge',8:'edge_only',9:'roof_edge',10:'sky_only',11:'component_only_ambiguous',12:'sky_and_edge',13:'sky_only',14:'sky_only',16:'pole_crossarm',17:'sky_only',19:'sky_only'}
    decisions=[]
    for row in manifest['rows']:
        idx=int(row['candidate_id'].split('-')[-1])
        status='hold_component_ambiguity' if idx in held else 'accepted_non_target_background_crop'
        decisions.append({**row,'semantic_review_status':status,'background_subtype':subtype[idx],
                          'reviewer':'assistant','review_note':'No four-class target device visible in crop.' if idx not in held else 'Only a component-like fragment is visible; hold until independent review.'})
    out={'schema_version':1,'source_manifest':str(source),'source_manifest_sha256':file_sha256(source),
         'rows':decisions,'accepted_count':sum(r['semantic_review_status'].startswith('accepted') for r in decisions),
         'held_count':sum(r['semantic_review_status'].startswith('hold') for r in decisions),
         'review_method':'Assistant inspected all 15 crops at high detail after contact-sheet review; not a blinded independent annotation.',
         'training_admitted':False,'requires_separate_training_gate':True,
         'limits':['Accepted crops are region-level negatives only; source frames contain positive targets.',
                   'Semantic review does not establish population false-positive rate.',
                   'Held component crop remains excluded until independently resolved.',
                   'No target class, historical label or protected evaluation member was changed.']}
    out['identity']=object_sha256(out);write_json(base/'semantic-review.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows',)},indent=2))


if __name__=='__main__':main()
