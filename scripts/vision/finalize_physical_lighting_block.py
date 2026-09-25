"""Bind explicitly authored blocking review, regressions and baseline; never train."""
import subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.run_physical_lighting_capture_v2 import OUT, prior, base
from scripts.vision.test_body_material_applicability import baseline_verify


def main():
    ep=OUT/'blocked-evidence/evidence.json'; e=prior.read(ep); prior.verify(e)
    cp=OUT/'capture-receipt.json'; c=prior.read(cp); prior.verify(c)
    for result in c['results']: prior.verify(prior.read(result['receipt']))
    if len(c['results'])!=5 or c['status']!='capture_blocked': raise ValueError('Unexpected stage state')
    for row in e['records']:
        if not row['rgb_exact'] or not row['pose_pass'] or row['max_rgb_skew_ms']>33.334: raise ValueError('Alignment changed')
        if [x['label'] for x in row['added']]!=['64']: raise ValueError('Evidence changed')
    report=prior.ROOT/'docs/results/ml_physical_lighting_block_20260910.md'
    review=OUT/'blocked-evidence/explicit-review.json'
    if not review.exists():
        prior.frozen(review,dict(review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            status='实例有可见证据，但内容不足',target='transformer_mid',runtime_label='64',
            reason='已查看全景、原尺寸ROI及实例叠图：图缘窄侧面与底部平台可见，主体严重截断；组件ID未知。实例归属由对齐掩码与保存映射确认，不依据外形猜测。历史全图标签未覆盖该实例，阻断沿用冻结曝光方案。',
            component_identity='unknown',training_ready=False,inputs={str(x):prior.file_sha256(x) for x in [ep,report,*sorted(ep.parent.glob('*.png'))]}))
    prior.verify(prior.read(review))
    tests=['tests.test_physical_lighting_preflight','tests.test_physical_lighting_capture','tests.test_condition_coverage','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_training_fit']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if t.returncode: raise ValueError(t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40: raise ValueError('Baseline changed')
    base.guard()
    if not all(prior.read(x['receipt'])['process_cleanup_complete'] for x in c['results']): raise ValueError('Cleanup incomplete')
    paths=[ep,cp,review,report,Path(__file__).resolve(),prior.ROOT/'config/perception/visual_experiment_baseline_v1.json']
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'blocked-completion.json',dict(status='stopped_full_label_membership_conflict',original_replays=5,
        original_technical_pass=4,lighting_captured=0,training_ready=False,training_started=False,
        blocker='L05 historical labels omit aligned visible transformer_mid; independent risk handling and revised freeze required',
        baseline=baseline,regression=dict(output=t.stderr,whole_repository_tested=False),process_cleanup_complete=True,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('BLOCK_CONFIRMED; 36 REGRESSIONS PASS; BASELINE 40 PASS; NO TRAINING')


if __name__=='__main__': main()
