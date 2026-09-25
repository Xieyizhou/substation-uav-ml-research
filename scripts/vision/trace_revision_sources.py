"""Bounded post-hoc source tracing and evidence, never modifies labels."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from scripts.vision.freeze_annotation_revision_design import OUT as DESIGN, read, verify, frozen, file_sha256, ROOT
from scripts.vision.test_body_material_applicability import source_index, model_mapping, equal_rgb, label_correspondence
from scripts.vision.structure_fit import truth_for
from scripts.vision.record_condition_target_review import normalize_mapping

OUT = DESIGN / 'source-review-v1'


def main():
    dest = OUT / 'evidence.json'
    if dest.exists():
        verify(read(dest)); print('VALID_SOURCE_EVIDENCE_REUSED'); return
    prior = DESIGN / 'design-receipt.json'; verify(read(prior))
    sources, paths = source_index(); paths += [prior, Path(__file__)]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for group in read(prior)['affected_source_groups']:
        lineage = group['source_lineage_id']
        closure = sorted(k for k, v in sources.items() if v.get('derivation_group') == lineage)
        for member in group['members']:
            mid = member['member_id']; src = sources[mid]; ip = Path(src['image_path'])
            if src.get('derivation_group') != lineage: raise ValueError('Source lineage mismatch')
            if file_sha256(ip) != src['image_sha256']: raise ValueError('Source image changed')
            for kind in ('image', 'label'):
                if file_sha256(Path(member[kind+'_path'])) != member[kind+'_sha256']: raise ValueError('Pool input changed')
            equal_rgb(ip, member['image_path'])
            cp = ip.parents[1] / 'collection-receipt.json'; pp = ip.parents[2] / 'plan/plan.json'; wp = pp.parent / 'world.sdf'
            receipt, plan = read(cp), read(pp)
            if plan['files']['world.sdf'] != file_sha256(wp): raise ValueError('World hash mismatch')
            if receipt.get('world_sha256') not in (None, file_sha256(wp)): raise ValueError('Receipt world changed')
            if receipt.get('plan_identity') not in (None, plan['identity']): raise ValueError('Plan identity changed')
            views = [v for v in receipt['views'] if v['view_id'] == ip.parent.name]
            if len(views) != 1 or views[0]['image_sha256'] != src['image_sha256']: raise ValueError('View identity mismatch')
            view = views[0]; mapping, _ = model_mapping(ET.parse(wp))
            rm = normalize_mapping(receipt.get('collection_checks', {}).get('instance_mapping', {}))
            if len({v['object_id'] for v in rm.values()}) != len(rm): raise ValueError('Mapping collision')
            for k, v in rm.items():
                if mapping.get(k) != v['object_id']: raise ValueError('Receipt mapping mismatch')
            truth = truth_for(member); original = label_correspondence(truth, view['truth']['objects'])
            labels = []
            for t in original:
                m = re.search(r'instance-(\d+)-', t['annotation_id'])
                label = str(int(m[1])) if m else None
                if label not in mapping: raise ValueError('Unresolved annotation instance')
                labels.append(dict(t, runtime_label=label, object_id=mapping[label]))
            target = '118' if group['review_id'] == 'T036' else '128'
            box = next((r['bbox_xyxy'] for r in labels if r['runtime_label'] == target), [0, 1030, 380, 1080])
            im = Image.open(ip).convert('RGB'); overlay = im.copy(); draw = ImageDraw.Draw(overlay)
            for row in labels:
                draw.rectangle(row['bbox_xyxy'], outline='red', width=3)
                draw.text(tuple(row['bbox_xyxy'][:2]), row['runtime_label'], fill='yellow')
            x1,y1,x2,y2 = box; roi = [max(0,int(x1)-30),max(0,int(y1)-30),min(im.width,int(x2)+30),min(im.height,int(y2)+30)]
            crop = im.crop(roi); crop.thumbnail((1100,650))
            if crop.height < 200: crop = crop.resize((crop.width*2,crop.height*2))
            canvas = Image.new('RGB', (1200, 1300), 'white')
            overlay.thumbnail((1180,680)); canvas.paste(overlay,(10,35)); canvas.paste(crop,(10,730))
            d = ImageDraw.Draw(canvas); d.text((10,10), f"{group['review_id']} {src.get('variant','regular')} | target {target} | red = existing labels", fill='black')
            page = OUT / (mid.split(':')[1][:12]+'.png'); canvas.save(page)
            paths += [ip, Path(member['image_path']), Path(member['label_path']),cp,pp,wp,page]
            results.append(dict(member_id=mid, review_id=group['review_id'], variant=src.get('variant','regular'),
                lineage_id=lineage, registered_source_members=closure, original_rgb_exact=True,
                complete_labels_correspond=True, labels=labels, target_runtime_label=target,
                target_in_saved_labels=target in [r['runtime_label'] for r in labels],
                target_world_object=mapping.get(target), source_receipt=str(cp),source_world=str(wp),
                actual_pose=view.get('actual_pose'), actual_annotation_mode=receipt.get('actual_annotation_mode'),
                post_hoc_only=True, historical_new_gate_pass_claimed=False,
                gaps=[k for k in ('world_sha256','actual_annotation_mode') if not receipt.get(k)] + ([] if rm else ['historical_instance_mapping_missing']),
                page=str(page), roi=roi, planned_exposures=member['planned_exposures'],actual_exposures_verified=False))
    frozen(dest,dict(status='source_evidence_ready_for_explicit_review', members=results,
        closure_scope='Four known training/development source ledgers only; not global independent-scene closure.',
        training_ready=False, training_started=False, historical_labels_changed=False,
        inputs={str(p):file_sha256(p) for p in paths}))
    print('FOUR_MEMBER_SOURCE_EVIDENCE_WRITTEN')


if __name__ == '__main__': main()
