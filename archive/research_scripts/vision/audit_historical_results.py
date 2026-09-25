"""Bounded historical development audit. No training, relabeling or inference."""
import argparse,json,re
from collections import Counter
from pathlib import Path
from src.ml.artifacts import object_sha256
from scripts.vision.train_unified_lighting import OUT as CURRENT,prior
from scripts.vision.unified_hold_lighting_counts import SOURCE
from scripts.vision import brightness_lr_retention as lr,lineage_capped_control as lc,prepare_whole_image_hold as wh
from scripts.vision.exposure_order_retention import PRIOR
from scripts.vision.run_fixed_budget_diagnosis import paired_truth,score,summary,VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision.verify_experiment_baseline import verify as baseline_verify

OUT=prior.ROOT/'data/research/ml_training_recovery_v1/historical-evidence-audit-v1'
SEEDS=(7,17,27)
def allowed(path):
    return not re.search(r'(^|[/_-])(sealed|protected|holdout|unseen)([/_.-]|$)',str(path),re.I)

def integrity(path,cache):
    if not allowed(path):return ['excluded_protected_scope']
    r=prior.read(path);errors=[]
    if object_sha256({k:v for k,v in r.items() if k!='identity'})!=r.get('identity'):errors.append('record_identity_mismatch')
    for raw,digest in r.get('inputs',{}).items():
        if not allowed(raw):errors.append('excluded_dependency:'+raw);continue
        if raw not in cache:
            try:cache[raw]=prior.file_sha256(raw)
            except OSError:cache[raw]=None
        if cache[raw]!=digest:errors.append('input_missing_or_changed:'+raw)
    return errors

def specs():
    out=[]
    for family,root,stem in [('R-clean',CURRENT,'R-clean'),('L-physical',CURRENT,'L-physical'),
      ('reviewed-hold',SOURCE,'brightness-450'),('brightness-lr0005',lr.OUT,'brightness-450'),
      ('brightness-lr001',lr.SOURCE,'brightness-450'),('lineage-capped',lc.OUT,'lineage-capped-450'),
      ('whole-reference',wh.OUT,'reference-450'),('whole-hold',wh.OUT,'hold-450')]:
        for seed in SEEDS:
            key=f'{stem}-{seed}';out.append(dict(family=family,seed=seed,key=key,evaluation=str(root/'evaluation'/f'{key}.json'),protocol=str(root/'protocol.json'),training=str(root/'training'/key/'completion.json')))
    for arm in ('retained_reference','retained_appearance'):
        for seed in SEEDS:out.append(dict(family=arm,seed=seed,key=f'{arm}-450-{seed}',evaluation=str(PRIOR/f'evaluation-{arm}-450-{seed}.json'),protocol=str(PRIOR/'protocol.json'),training=None))
    old=prior.ROOT/'data/research/ml_training_recovery_v1/exposure-controlled-diagnosis-v1'
    for seed in SEEDS:out.append(dict(family='historical-A',seed=seed,key=f'historical-A-{seed}',evaluation=str(old/'evaluation'/f'historical-A-{seed}.json'),protocol=str(old/'protocol.json'),training=None))
    return out

def remeasure(r,pairs,neg):
    rows=r['rows'];nr=r['negative_rows']
    if len(rows)!=48 or len(nr)!=48:raise ValueError('Incomplete 48+48 evaluation')
    if {(x['view_id'],x['variant']) for x in rows}!=set(pairs) or {(x['view_id'],x['variant']) for x in nr}!=set(neg):raise ValueError('Membership mismatch')
    fresh=[];row_diffs=[]
    for x in rows:
        source,truth=pairs[x['view_id'],x['variant']]
        if source['image_sha256']!=x['image_sha256'] or truth!=x['truth']:raise ValueError('Truth/image differs from current frozen reference')
        y=score(source,truth,x['predictions'],x['low_predictions']);fresh.append(y)
        if y!=x:row_diffs.append((x['view_id'],x['variant']))
    for x in nr:
        if x['image_sha256']!=neg[x['view_id'],x['variant']]['image_sha256']:raise ValueError('Negative image mismatch')
    sm={v:summary([x for x in fresh if x['variant']==v]) for v in VARIANTS}
    ns=dict(frame_false_positive_rate=sum(bool(x['predictions']) for x in nr)/48,unmatched_predictions=sum(len(x['predictions']) for x in nr))
    delta={v:{m:sm[v][m]-r['summary'][v][m] for m in ('planned_instance_hit_rate','instance_recall','matched_precision','unmatched_predictions')} for v in VARIANTS}
    return dict(summary=sm,negative_summary=ns,matching_conflicts=sum(x['matching_conflict'] for x in fresh),row_difference_count=len(row_diffs),
        stored_summary_equal=sm==r['summary'],stored_negative_equal=ns==r['negative_summary'],metric_deltas=delta)

def freeze():
    path=OUT/'protocol.json'
    if path.exists():prior.verify(prior.read(path));return prior.read(path)
    p=prior.read(CURRENT/'protocol.json');docs=sorted((prior.ROOT/'docs/results').glob('ml_*.md'))
    inventory=[dict(path=str(x),sha256=prior.file_sha256(x),status='catalogued_not_claim_verified') for x in docs]
    pairs=[CURRENT/'protocol.json',Path(__file__).resolve()]
    return prior.frozen(path,dict(status='audit_scope_frozen',cells=specs(),report_inventory=inventory,evaluation=p['evaluation'],
        scope='All ml result reports catalogued; eleven decision-critical families/33 cells rescored. Earlier claims outside these families not declared validated.',
        excluded=['protected labels','sealed scene tests','new inference','training','historical mutation','automatic label correction'],
        inputs={str(x):prior.file_sha256(x) for x in pairs}))

def main():
    p=freeze();cache={};paths=[OUT/'protocol.json'];base=prior.read(CURRENT/'protocol.json')
    rp,np=Path(p['evaluation']['paired_review']),Path(p['evaluation']['negative_review'])
    for x in (rp,np):
        err=integrity(x,cache)
        if err:raise ValueError('Current development reference invalid: '+str(err))
    paired,truthinputs=paired_truth(prior.read(rp)['frames']);pairs={(r['view_id'],r['variant']):(r,t) for r,t in paired}
    neg={(r['view_id'],r['variant']):r for r in prior.read(np)['frames']};paths += [rp,np]
    for r in [r for r,_ in paired]+list(neg.values()):
        if not allowed(r['image_path']) or prior.file_sha256(r['image_path'])!=r['image_sha256']:raise ValueError('Development RGB changed')
        paths.append(Path(r['image_path']))
    results=[];groups={}
    for spec in p['cells']:
        item=dict(**spec,issues=[])
        try:
            ep=Path(spec['evaluation']);r=prior.read(ep);paths.append(ep);item['issues']+=integrity(ep,cache)
            pp=Path(spec['protocol']);q=prior.read(pp);paths.append(pp);item['issues']+=integrity(pp,cache)
            item['initialization']=q.get('initialization',q.get('controls',{}).get('initial_weights'))
            if spec['training']:
                cp=Path(spec['training']);c=prior.read(cp);paths.append(cp);item['issues']+=integrity(cp,cache)
                wp=Path(c['weights']);item['terminal_weight_valid']=prior.file_sha256(wp)==c['weights_sha256'];paths.append(wp)
                xp=Path(c['exposure_path']);x=prior.read(xp);paths.append(xp);item['issues']+=integrity(xp,cache)
                item['actual_exposure_matches_plan']=x['draws']==q['schedules'][spec['key']]
                rows={z['member_id']:z for z in q['pool_rows']};active=set(x['draws'])
                item['unique_members']=len(active);item['registered_lineages']=len({str(rows[m].get('lineage_id','unknown')) for m in active})
                item['lineage_count_is_not_independent_scenes']=True
                item['optimizer_steps']=c.get('optimizer_steps')
                if not item['terminal_weight_valid'] or not item['actual_exposure_matches_plan']:item['issues'].append('weight_or_exposure_failed')
            else:item['training_provenance_scope']='not_revalidated_in_this_pass'
            item['remeasured']=remeasure(r,pairs,neg);groups.setdefault(spec['family'],[]).append(dict(**item['remeasured']))
            item['status']='numerically_recomputed_with_identity_gaps' if item['issues'] else 'numerically_recomputed_identity_checked'
        except (OSError,ValueError,KeyError,TypeError) as exc:item['status']='blocked';item['issues'].append(str(exc))
        results.append(item);print('AUDITED',spec['family'],spec['seed'],item['status'],flush=True)
    agg={g:aggregate(rs) for g,rs in groups.items() if len(rs)==3}
    gates={g:policy_checks(a,agg['retained_reference'],agg['historical-A'],base) for g,a in agg.items()} if 'retained_reference' in agg and 'historical-A' in agg else {}
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    issuecount=Counter(x['status'] for x in results)
    paths += [Path(x) for x in truthinputs]
    prior.frozen(OUT/'results.json',dict(status='bounded_historical_audit_complete_with_scope_limits',cells=results,aggregate=agg,recomputed_gates=gates,
        counts=dict(issuecount),baseline=baseline,global_water_inflation_percentage=None,
        current_data_independence=dict(paired_images=48,paired_poses=12,negative_images=48,negative_pose_groups=len({x['view_id'] for x in neg.values()}),independent_scenes='not_established'),
        interpretation='Zero arithmetic delta does not certify label correctness, causal isolation or generalization. No label-risk sensitivity estimate without independent confirmed corrections.',
        inputs={str(x):prior.file_sha256(x) for x in set(paths) if x.exists() and allowed(x)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit',action='store_true');args=ap.parse_args()
    if args.audit:main()
    else:freeze();print('SCOPE_ONLY_NO_TRAINING_NO_INFERENCE')
