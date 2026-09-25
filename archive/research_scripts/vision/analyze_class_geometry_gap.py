"""Aggregate independent development results by class, map and target scale."""
import json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json

def bucket(box):
 a,b,c,d=box;area=max(0,c-a)*max(0,d-b);height=max(0,d-b)
 return 'small' if area<40000 else 'medium' if area<180000 else 'large'

def summarize(rows,weight):
 out=defaultdict(lambda:{'frames':0,'observable_expected_frames':0,'expected_absent_frames':0,'expected_hits':0,'matched_hits':0,'truth_objects':0,'predictions':0,'by_scale':defaultdict(lambda:{'frames':0,'expected_hits':0,'truth_objects':0})})
 for row in rows:
  cat=row.get('expected') or row.get('category') or row.get('source',{}).get('expected_category');truth=row.get('truth') or row.get('source',{}).get('truth',[])
  if 'score' in row: score=row['score'];expected_hit=score['expected_class_hit'];matched=score['matched_hits'];pred=score['predictions']
  else: expected_hit=row['expected_hit'];matched=row['hits'];pred=row['predictions']
  expected_present=row.get('expected_class_present', row.get('score',{}).get('expected_class_present', True))
  group=out[(row.get('map_id') or 'simple-independent',cat)];group['frames']+=1;group['observable_expected_frames']+=int(bool(expected_present));group['expected_absent_frames']+=int(not expected_present);group['expected_hits']+=int(bool(expected_present and expected_hit));group['matched_hits']+=matched;group['truth_objects']+=len(truth);group['predictions']+=pred
  target=[o for o in truth if o['class_name']==cat]
  if target:
   s=group['by_scale'][bucket(target[0]['bbox_xyxy'])];s['frames']+=1;s['expected_hits']+=int(bool(expected_present and expected_hit));s['truth_objects']+=len(target)
 return {f'{k[0]}:{k[1]}':{**v,'by_scale':dict(v['by_scale'])} for k,v in out.items()}

def main():
 base=ROOT/'data/research/ml_training_recovery_v1';inputs={};all_results={}
 cross=json.loads((base/'cross-scene-recheck-v1/evaluation.json').read_text());inputs[str(base/'cross-scene-recheck-v1/evaluation.json')]=file_sha256(base/'cross-scene-recheck-v1/evaluation.json')
 for weight,result in cross['results'].items():
  rows=[r for r in result['rows'] if not r['prior_diagnostic_pixel_overlap']];all_results.setdefault(weight,[]).extend(rows)
 simple=json.loads((base/'simple-independent-scene-v1/evaluation.json').read_text());inputs[str(base/'simple-independent-scene-v1/evaluation.json')]=file_sha256(base/'simple-independent-scene-v1/evaluation.json')
 for weight,result in simple['results'].items():all_results.setdefault(weight,[]).extend(result['rows'])
 matrix=json.loads((base/'negative-ab-v1/independent-recheck.json').read_text());inputs[str(base/'negative-ab-v1/independent-recheck.json')]=file_sha256(base/'negative-ab-v1/independent-recheck.json')
 for weight,result in matrix['results'].items():
  rows=[r for r in result['groups']['matrix_controls']]
  for r in rows:
   # Reformat the matrix source/score pair into the same aggregation shape.
   all_results.setdefault(weight,[]).append({'map_id':r['map_id'],'expected':r['expected'],'truth':r['truth'],'score':r['score']})
 report={'inputs':inputs,'status':'class_geometry_gap_analysis_complete','results':{w:summarize(rows,w) for w,rows in all_results.items()},'training_admitted':False,'limits':['Development-only observations; no protected validation.','Scale is image-space box area, not distance or physical size.','Matched hits and expected-class hits use the existing IoU 0.5 protocol.','Expected-class hits are counted only when the expected class is present in full_2d truth; expected-absent rows are scene/visibility or annotation holds.','Cabinet/empty-ground rows are not target recall measures.']}
 report['identity']=object_sha256(report);write_json(base/'class-geometry-gap-v1.json',report)
 print(json.dumps(report['results'],indent=2))
if __name__=='__main__':main()
