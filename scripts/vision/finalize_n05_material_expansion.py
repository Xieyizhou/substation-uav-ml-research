"""Record explicit observations, quarantine lineage, and stop unsafe expansion."""
from datetime import datetime,timezone
from pathlib import Path
import copy
import subprocess,sys
from scripts.vision.expand_material_view_n05 import OUT,SOURCE,prior
from scripts.vision.evaluate_scale_endpoints import baseline_verify

NOTES={
 'original-00':'后排蓝色柜体面板、顶面和侧面可辨，左下被邻柜遮挡。',
 'original-01':'后排蓝色柜体宽侧面、面板及基座可辨，左侧部分遮挡。',
 'original-02':'中间蓝色柜体主体、前面板、顶面和基座清晰。',
 'original-03':'左前景蓝色柜体宽背面、顶面和基座可辨，左下图缘截断。',
 'original-04':'右侧蓝色柜体前面板、主体及基座可辨，右图缘截断。',
 'warm-00':'后排蓝色柜体主体及窄面板可见，仍受邻柜遮挡。',
 'warm-01':'蓝色后排柜体宽侧面和面板可辨，前景暖色邻柜遮挡左部。',
 'warm-02':'计划柜体变为暖灰棕色，面板保持深色，主体和顶面完整清楚。',
 'warm-03':'左前景蓝色宽背面及顶面可辨，图缘截断未改变。',
 'warm-04':'右侧蓝色柜体面板和基座清楚，右缘截断。',
 'cool-00':'后排蓝色柜体主体、顶面和面板部分可辨，遮挡仍在。',
 'cool-01':'蓝色后排柜体宽侧面、面板及基座可辨，灰色前景邻柜局部遮挡。',
 'cool-02':'计划柜体变为冷灰色，深色面板、主体、顶面与基座清晰可辨。',
 'cool-03':'左前景蓝色柜体背面和顶面可辨，左下截断。',
 'cool-04':'右侧蓝色柜体面板和主体可辨，右缘截断。',
}


def main():
    ep=OUT/'review/evidence.json';ap=OUT/'coverage-audit/audit.json'
    evidence=prior.read(ep);audit=prior.read(ap);prior.verify(evidence)
    # Narrow repair for this newly generated artifact: original hashing used integer keys,
    # while JSON restored strings. Reconstruct exactly and verify before publishing v2.
    restored=copy.deepcopy(audit)
    for row in restored['records']:
        row['visible_pixels_by_label']={int(k):v for k,v in row['visible_pixels_by_label'].items()}
    prior.verify(restored)
    original_audit=ap;ap=OUT/'coverage-audit/audit-v2.json'
    if not ap.exists():
        prior.frozen(ap,dict(records=audit['records'],status=audit['status'],training_ready=False,labels_modified=False,
            correction='Integer-key hashing versus JSON string-key roundtrip; reconstructed original verified before creating string-key version. Counts and decisions unchanged.',
            inputs={**audit['inputs'],str(original_audit):prior.file_sha256(original_audit),str(Path(__file__).resolve()):prior.file_sha256(Path(__file__))}))
    audit=prior.read(ap);prior.verify(audit)
    if len(evidence['events'])!=15 or {x['event_id'] for x in evidence['events']}!=set(NOTES):raise ValueError('Review coverage mismatch')
    now=datetime.now(timezone.utc).isoformat()
    decisions=[dict(e,reason=NOTES[e['event_id']],reviewed_at=now,review_type='AI辅助审核',
        status='content_observed_not_admitted',whole_image_status='held_unboxed_transformer',training_admitted=False,promotable=False) for e in evidence['events']]
    review=prior.read(SOURCE/'source-review.json');inventory=prior.read(SOURCE/'source-inventory.json')
    groups={x['lineage_id'] for x in review['sources'] if x['source_review_id'] in ('N04','N05')}
    denied=[x['source_member_id'] for x in inventory['records'] if x.get('lineage_id') in groups]
    tests=['tests.test_material_mask_coverage','tests.test_material_expansion_gate','tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths=[ep,ap,OUT/'completion.json',OUT/'protocol.json',SOURCE/'source-review.json',SOURCE/'source-inventory.json',Path(__file__),
        prior.ROOT/'docs/results/ml_material_expansion_risk_20260911.md']
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'reviewed-completion.json',dict(status='risk_contained_expansion_stopped_at_verified_coverage_gap',decisions=decisions,
        unboxed_target_review=dict(object_id='transformer_sw',runtime_label=76,visible_pixels=19117,bbox_xyxy=[1747,722,1920,1080],
            review_type='AI辅助审核',reviewed_at=now,reason='右下图缘窄侧面及基座片段；叠图显示属于实例76，但原完整框输出中无该实例。实例存在性已确认，内容不足风险同时保留；不自动补框。'),
        denied_lineages=sorted(groups),denied_registered_source_members=sorted(denied),
        held_derivatives=['N04 original/warm/cool','N05 original/warm/cool'],
        quarantine_effect='New independent deny ledger for all registered same-pose variants; no old file deleted or changed. Must be consumed by any subsequent candidate export.',
        new_images_this_turn=3,total_new_material_images=6,admitted_images=0,training_started=False,training_ready=False,
        baseline=b,regression_output=t.stdout+t.stderr,whole_repository_tests_claimed=False,
        next_priority='Establish fresh source views with complete per-frame mask-to-box coverage before generating appearance variants. Keep old data held; independent label repair only as separately specified work.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('REVIEW15; HELD_N04_N05; TEST26_PASS; PINNED40_PASS; NO_TRAINING',len(denied))

if __name__=='__main__':main()
