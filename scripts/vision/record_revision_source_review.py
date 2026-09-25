"""Explicit four-target observations transcribed after viewing the evidence pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.trace_revision_sources import OUT, ROOT, read, verify, frozen, file_sha256
from scripts.vision.test_body_material_applicability import baseline_verify

EVIDENCE = '815a8f5260dab4fdc80b83a02eb790dba128f71a8036a83c13564deb98cccab3'
OBSERVATIONS = {
 'regular': '前景变压器占据框内大部分面积；后方仅见窄顶部、右侧平面与底部条带，不能据此批准完整设备监督。',
 'original': '底图缘可见青色尖窄平面片段，主体其余部分在画面之外；本帧尚无独立掩码认证，实例归属保持待核验。',
 'neutral_bridge': '底图缘仅见灰色尖窄平面片段，缺少足够主体与面板内容；此前同帧重放支持实例存在，不等于监督可用。',
 'background_bridge': '底图缘可见青色尖窄平面片段，背景变化未改善截断；本帧尚无独立掩码认证，实例归属保持待核验。',
}


def validate(decisions, evidence):
    expected = {r['member_id'] for r in evidence['members']}
    ids = [r['member_id'] for r in decisions]
    if len(ids) != len(set(ids)) or set(ids) != expected: raise ValueError('Missing or duplicate decision')
    for d in decisions:
        if d['evidence_identity'] != evidence['identity']: raise ValueError('Stale evidence identity')
        if file_sha256(Path(d['page'])) != d['page_sha256']: raise ValueError('Stale page')
        if d['decision'] != 'hold_pending' or not d['reason']: raise ValueError('Unexpected review decision')


def main():
    ep = OUT / 'evidence.json'; evidence = read(ep); verify(evidence)
    if evidence['identity'] != EVIDENCE: raise ValueError('Reinspection required')
    paths = [ep, Path(__file__), ROOT/'docs/results/ml_revision_source_review_20260910.md']
    decisions = []
    for r in evidence['members']:
        page = Path(r['page']); paths.append(page)
        decisions.append(dict(member_id=r['member_id'],target_runtime_label=r['target_runtime_label'],
            decision='hold_pending', reason=OBSERVATIONS[r['variant']], review_nature='AI-assisted',
            reviewed_at=datetime.now(timezone.utc).isoformat(), evidence_identity=EVIDENCE,
            page=str(page),page_sha256=file_sha256(page),
            scope='Named risk target and full-frame context only; not approval of all existing labels.',
            original_pixel_instance_certified_this_run=False, training_admitted=False,promotable=False))
    validate(decisions,evidence)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40: raise ValueError('Baseline failed')
    dest=OUT/'review.json'
    if dest.exists(): verify(read(dest)); print('VALID_REVIEW_REUSED'); return
    frozen(dest,dict(status='bounded_source_trace_complete_four_risk_targets_held',decisions=decisions,
        baseline=baseline,training_ready=False,training_started=False,historical_labels_changed=False,
        unresolved=['Original/background T027 require separate aligned instance evidence before attribution certification.',
            'Extreme truncation/occlusion supervision suitability remains unresolved.',
            'Actual training exposure receipts not audited in this source-only step.',
            'Closure limited to four registered source ledgers; no global or independent-scene claim.'],
        inputs={str(p):file_sha256(p) for p in paths}))
    print('FOUR_EXPLICIT_HOLD_DECISIONS; BASELINE_40_PASSED')


if __name__ == '__main__': main()
