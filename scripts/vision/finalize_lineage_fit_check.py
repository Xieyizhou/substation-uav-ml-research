"""Explicit inspected findings and integrity-bound targeted diagnosis completion."""
import hashlib,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
from scripts.vision.check_lineage_training_fit import OUT,ROOT,SOURCE,read,verify,frozen,file_sha256,valid
from scripts.vision.prepare_whole_image_hold import baseline_verify

OBS={
 ('hold-450-17','C08-L4'):'右边界截断圆柱，曲面和基座可见；保持非清晰完整子集，不作像素级可见性认证。',
 ('lineage-capped-450-17','C04-L3'):'独立小矩形主体、顶面和基座清楚，无遮挡但视觉类别特征少；来源登记电容器组，不能因预测开关柜而改标签。',
 ('lineage-capped-450-17','C25-L7'):'圆柱顶面、连续侧面和基座完整，旁边杆体未遮主体；清晰子集归类仍适用。',
 ('reference-450-17','C30-L8'):'右图缘圆柱主体大部可见，底边和右侧被近处结构叠压；保留部分遮挡归类。',
 ('reference-450-17','C39-L7'):'右边界圆柱的下部被前景设备遮挡，框内包含前景顶面与接线柱；非清晰完整子集。',
}

def pixel_id(path):
    with Image.open(path) as im:
        im=im.convert('RGB');return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def main():
    p=read(OUT/'protocol.json');s=read(OUT/'summary.json');f=read(OUT/'findings.json')
    for r in (p,s,f):verify(r)
    if set(OBS)!={(e['detail']['model'],e['detail']['event_id']) for e in f['events']}:raise ValueError('Explicit inspection missing/duplicate')
    decisions=[];now=datetime.now(timezone.utc).isoformat()
    for e in f['events']:
        key=(e['detail']['model'],e['detail']['event_id'])
        decisions.append(dict(model=key[0],event_id=key[1],review_nature='AI辅助审核',reviewed_at=now,reason=OBS[key],
            source_image_sha256=e['source']['image_sha256'],source_label_sha256=e['source']['label_sha256'],truth=e['truth'],
            evidence_sha256=e['evidence_sha256'],detail=e['detail'],pixel_visibility_certified=False,training_approved=False))
    review=OUT/'inspection.json'
    if review.exists():verify(read(review))
    else:frozen(review,dict(status='five_explicit_fit_miss_observations',decisions=decisions,inputs={str(x):file_sha256(x) for x in [OUT/'findings.json',Path(__file__)]}))
    for key in p['models']:valid(read(OUT/'inference'/f'{key}.json'),key,p)
    # Compare original-size RGB rather than compressed file identity only.
    train={r['member_id']:pixel_id(r['image_path']) for r in p['rows']}
    source=read(SOURCE/'protocol.json');dev=read(source['evaluation']['paired_review']);verify(dev)
    dev_pixels=[(r,pixel_id(r['image_path'])) for r in dev['frames']]
    overlap=[dict(member_id=m,view_id=r['view_id'],variant=r['variant']) for r,rgb in dev_pixels for m,h in train.items() if rgb==h]
    if overlap:raise ValueError('Exact train/development pixels overlap')
    tests=['tests.test_lineage_training_fit','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_capped_results']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Pinned40 failure')
    paths=[OUT/'protocol.json',OUT/'summary.json',OUT/'findings.json',review,Path(__file__),ROOT/'scripts/vision/check_lineage_training_fit_v2.py',ROOT/'docs/results/ml_reviewed_target_fit_20260910.md']+[ROOT/(t.replace('.','/')+'.py') for t in tests]
    frozen(OUT/'completion.json',dict(status='targeted_diagnosis_complete_no_training',inference_units=9,images=47,reviewed_targets=49,clear_targets=26,
        explicit_current_miss_inspections=5,regression_output=result.stderr,baseline=baseline,whole_repository_tested=False,
        original_size_RGB_overlap_with_paired_development=overlap,training_pixel_ids=train,
        priority='Controlled development-condition transfer diagnosis; retain two seed17 training-fit regressions as guard cases.',
        scope_limit='Reviewed subset only; no independent scene count inferred and no unique root cause established.',inputs={str(x):file_sha256(x) for x in paths}))
    print('COMPLETE_9_INFERENCE_UNITS_27_TESTS_PINNED40_NO_TRAINING')

if __name__=='__main__':main()
