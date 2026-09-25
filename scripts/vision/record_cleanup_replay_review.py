"""Explicit inspection of 27 aligned instance overlays and conservative whole-frame holds."""
from datetime import datetime,timezone
from scripts.vision.run_visibility_cleanup_validation import OUT,ROOT,prepare,read,save,file_sha256,verify_tree,Path
from scripts.vision.record_held_visibility_followup import OBS
from scripts.vision.finalize_instance_visibility import validate_reviews

DETAILS={
'T049':'实例掩码仅覆盖右侧很小的基部/边缘，不能提供完整类别外形。',
'T054':'掩码仅覆盖左边缘基座薄片，不能按空标签删除，也不足以辨识主体。',
'T055':'可见目标仅为右上窄片，下方大面积青色箱体不属于该实例。',
'T077':'目标掩码在后方窄条；前方小暗面板不属于该实例，类别内容不足。',
'T095':'目标仅为立柱后方的窄顶面/上缘，前景大箱体不属于该实例。',
'T111':'仅有后方箱体上段短侧面，前景顶面与立柱不属目标，关键轮廓被遮住。',
'T116':'后方宽但矮的箱体上段可见，前景大立面不属目标，保守暂缓整图。',
'T117':'目标只剩立柱后方的小片上段，前景大面不属于目标。',
'T124':'仅49个原图像素在右上角，无法形成可辨识类别内容。',
'T131':'目标仅露出后方顶部窄片；前方深色面板和立杆不属于目标。',
'T148':'右边缘仅有小块基座，主体不在画面内。',
'T159':'左边缘较长斜侧面属于目标，但主体被截断并与前景重叠，保守暂缓。',
'T175':'掩码仅在后方顶部细条，前方带暗面板箱体并非目标实例。',
'T178':'掩码明确圈出后方箱体，顶部及相邻立面形成可辨识连续外形，前景遮挡边界可区分；保留其原有成员身份，不把本审核当作全标签重新准入。',
'T187':'左侧仅露后方短侧面，前景顶面遮住其余轮廓，保守暂缓。',
'T236':'仅有三根前景立柱后方的细条属于目标。',
'T241':'灰色条件下同样只有立柱后方细条属于目标，不把前景顶面算入。',
'T246':'目标仅剩后方细条，前景顶面及立柱均不属于目标。',
'T269':'目标掩码为后方极薄上沿，前景大箱体不是该实例。',
'T270':'目标为前景顶面后方窄带，立柱不属于目标。',
'T274':'灰色变体目标仍为后方极薄上沿，几何可见区域与同源变体一致。',
'T275':'灰色变体目标为后方窄带，不含前景大面。',
'T279':'常规变体同样仅露极薄上沿，不能把前景大立面当目标。',
'T280':'常规变体仅有顶面后方窄带属于目标。',
'T284':'前景立柱和箱体遮住目标，只有右后方极小顶部片段。',
'T288':'灰色变体也只有后方小片顶部，不含前景大箱体。',
'T292':'常规变体后方小片顶部可见，不足以辨认完整类别轮廓。',
}

def main():
    p=prepare();mp=OUT/'review-manifest.json';verify_tree(mp);m=read(mp)
    if set(DETAILS)!={r['review_id'] for r in m['items']}:raise ValueError('Explicit review incomplete')
    decisions=[];inputs={str(q):file_sha256(q) for q in (mp,Path(__file__))}
    for row in m['items']:
        rid=row['review_id'];paths=[row['source_image'],row['label_path'],row['receipt_path'],row['crop_path'],row['page_path']]
        bindings={str(q):file_sha256(q) for q in paths};inputs.update(bindings)
        decisions.append(dict(review_id=rid,member_id=row['member_id'],status='visible_identifiable' if rid=='T178' else 'visible_content_insufficient',
            action='retain_prior_membership' if rid=='T178' else 'hold_whole_frame_and_derivation_group',
            original_frame_certified=True,visible_pixel_count=row['visible_pixel_count'],visible_bbox_xyxy=row['visible_bbox_xyxy'],component_evidence='unknown',
            review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),reason=DETAILS[rid],inputs=bindings))
    validate_reviews(p,decisions)
    held=sorted({d['member_id'] for d in decisions if d['action'].startswith('hold')})
    save(OUT/'review.json',dict(status='all_27_explicitly_reviewed',decisions=decisions,held_members=held,inputs=inputs,
        scope='Additional visibility audit. Whole-frame holds are conservative diagnostic filtering, not proof of erroneous annotation. No single-box deletion or empty-label conversion.',
        pixel_area_threshold=None))
    save(OUT/'completion.json',dict(status='cleanup_and_full_replay_verified',frames=23,events=27,held_members=len(held),
        all_runtime_exits_normal=True,original_labels_modified=False,inputs={str(OUT/'review.json'):file_sha256(OUT/'review.json')}))
    print('REVIEW_COMPLETE',len(decisions),'HELD_MEMBERS',len(held))

if __name__=='__main__':main()
