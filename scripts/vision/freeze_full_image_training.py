"""Export reviewed complete-frame labels and freeze a matched development contrast.

Never edits source labels, starts training, or grants formal admission.
"""
import math
import random
import traceback
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.capture_full_image_remaining import OUT as SOURCE, ORIGINAL, PRIOR, read, save, file_sha256, verify_tree
from scripts.vision.prepare_instance_exposure_balance import OUT as REFERENCE
from scripts.vision.exposure_protocol import ROOT, NAMES, SEEDS, exposures, cycle
from scripts.vision.finalize_full_image_remaining import validate_matrix
from scripts.vision.verify_experiment_baseline import verify as baseline_verify
from src.ml.artifacts import object_sha256

OUT = SOURCE / 'matched-appearance-training-v1'
VARIANTS = ('steel_original', 'steel_warm_dim', 'ochre_original', 'ochre_warm_dim', 'sage_original', 'sage_warm_dim')

def encode(objects, size):
    w, h = size
    if w <= 0 or h <= 0 or not objects:
        raise ValueError('Invalid image or empty positive annotation')
    if len({o['object_id'] for o in objects}) != len(objects):
        raise ValueError('Duplicate instance')
    result = []
    for obj in objects:
        if obj['review_status'] != 'visible_content_observed' or not obj.get('reason'):
            raise ValueError('Missing explicit instance review')
        x1, y1, x2, y2 = obj['bbox_xyxy']
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('Invalid box; never silently clip')
        values = ((x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h)
        line = str(NAMES.index(obj['category'])) + ' ' + ' '.join(f'{v:.12f}' for v in values)
        _, cx, cy, bw, bh = map(float, line.split())
        recovered = ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)
        if max(abs(a-b) for a,b in zip(recovered,(x1,y1,x2,y2))) > 1e-6:
            raise ValueError('Label roundtrip loss')
        result.append(line)
    return '\n'.join(result) + '\n'

def check_source(frame, row, mapping):
    if row['status'] != 'captured' or frame['image_sha256'] != row['image_sha256'] or frame['view_id'] != row['view_id']:
        raise ValueError('Capture identity mismatch')
    if object_sha256(row['raw_truth']) != frame['truth_sha256']:
        raise ValueError('Raw annotation changed')
    found = []
    for b in row['raw_truth'].get('annotatedBox', []):
        identity = mapping[str(b['label'])]
        lo, hi = b['box'].get('minCorner', {}), b['box'].get('maxCorner', {})
        found.append((identity['object_id'], identity['category'], b['label'],
                      [float(lo.get('x',0)),float(lo.get('y',0)),float(hi.get('x',0)),float(hi.get('y',0))]))
    observed = [(o['object_id'],o['category'],o['runtime_label'],o['bbox_xyxy']) for o in frame['objects']]
    if sorted(found) != sorted(observed):
        raise ValueError('Review does not preserve every raw instance')

def schedules(prior, members, seed):
    lookup = {r['member_id']:r for r in prior['pool_rows']}
    index = {(r['pair_id'],r['variant']):r['member_id'] for r in members}
    poses = sorted({r['pair_id'] for r in members})
    pose_stream = cycle(poses, 360, random.Random(f'full-image-poses:{seed}'))
    variant_streams = {p:cycle(VARIANTS, 45, random.Random(f'full-image-variant:{seed}:{p}')) for p in poses}
    offsets = Counter(); control = []; appearance = []; swaps = []
    for i, mid in enumerate(prior['schedules'][f'I-300-{seed}']):
        if lookup[mid]['subset'] == 'bridge_positive':
            pose = pose_stream[len(swaps)]; variant = variant_streams[pose][offsets[pose]]; offsets[pose] += 1
            a,b = index[pose,'original'],index[pose,variant]
            control.append(a); appearance.append(b)
            swaps.append(dict(position=i,pair_id=pose,control=a,appearance=b,variant=variant))
        else:
            control.append(mid); appearance.append(mid)
    if len(control) != 1800 or len(swaps) != 360 or set(offsets.values()) != {45}:
        raise ValueError('Unexpected reference quota')
    allrows = {r['member_id']:r for r in prior['pool_rows']+members}
    for a,b in zip(control,appearance):
        if allrows[a]['class_instances'] != allrows[b]['class_instances']:
            raise ValueError('Paired supervision counts differ')
    return control, appearance, swaps

def main():
    target = OUT/'protocol.json'
    if target.exists():
        verify_tree(target); print('VERIFIED_EXISTING', target); return
    paths = [SOURCE/'matrix-handoff.json', ORIGINAL/'review/decisions.json', PRIOR/'review/decisions.json',
             SOURCE/'review/decisions.json', REFERENCE/'protocol.json', REFERENCE/'completion.json',
             Path(__file__), ROOT/'tests/test_full_image_training_freeze.py']
    seen = set()
    for p in paths[:6]: verify_tree(p, seen)
    if read(paths[0])['status'] != '56_frame_matrix_reviewed_not_training_dataset_frozen':
        raise ValueError('Matrix not ready')
    originals = read(paths[1])['frames']; variants = read(paths[2])['frames'] + read(paths[3])['frames']
    validate_matrix(originals, variants)
    frames = originals + variants
    prior = read(REFERENCE/'protocol.json')
    OUT.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (OUT/f'attempt-{attempt:03}').exists(): attempt += 1
    run = OUT/f'attempt-{attempt:03}'; run.mkdir()
    try:
        (run/'images').mkdir(); (run/'labels').mkdir()
        inputs = {str(p):file_sha256(p) for p in paths}; members = []; receipts = {}
        for f in frames:
            if f['review_nature'] != 'AI辅助审核' or not f.get('reviewed_at'):
                raise ValueError('Missing review provenance')
            source = Path(f['image_path']); rp = source.parent.parent/'collection-receipt.json'
            if rp not in receipts:
                r = read(rp); receipts[rp] = r; inputs[str(rp)] = file_sha256(rp)
            r = receipts[rp]; row = next(x for x in r['views'] if x['view_id'] == f['view_id'])
            check_source(f,row,r['collection_checks']['instance_mapping'])
            if file_sha256(source) != f['image_sha256']: raise ValueError('Stale RGB')
            inputs[str(source)] = f['image_sha256']
            with Image.open(source) as src: image = src.convert('RGB')
            fid = f['frame_id']; ip = run/'images'/f'{fid}.png'; lp = run/'labels'/f'{fid}.txt'
            labels = encode(f['objects'], image.size)
            image.save(ip)
            with Image.open(ip) as exported:
                if exported.convert('RGB').tobytes() != image.tobytes(): raise ValueError('PNG pixels differ')
            lp.write_text(labels)
            inputs[str(ip)] = file_sha256(ip); inputs[str(lp)] = file_sha256(lp)
            members.append(dict(member_id='full-image:'+fid, image_path=str(ip), label_path=str(lp),
                image_sha256=file_sha256(ip), label_sha256=file_sha256(lp), source_image_path=str(source),
                source_image_sha256=f['image_sha256'], source_truth_sha256=f['truth_sha256'],
                width=image.width,height=image.height,pair_id=f.get('original_frame_id',fid),variant=f.get('variant','original'),
                lineage_id=f['shared_source_group'],lineage_resolution='shared_source_group',
                subset='appearance_paired',data_role='development_training_candidate',
                class_instances=dict(Counter(o['category'] for o in f['objects'])),objects=f['objects'],
                review_nature=f['review_nature'],reviewed_at=f['reviewed_at'],map_id=row['map_id'],
                source_view_id=f['view_id'],source_receipt=str(rp),training_admitted=False,promotable=False))
        if len(members)!=56 or sum(sum(r['class_instances'].values()) for r in members)!=77:
            raise ValueError('Export count mismatch')
        dataset = run/'dataset.json'
        save(dataset,dict(status='frozen_reviewed_development_candidates_not_formal_admission',members=members,
            frame_count=56,box_observations=77,source_groups=7,pose_count=8,map_count=1,
            pixel_conversion='lossless RGB PPM to PNG; exact original-resolution byte comparison',inputs=inputs))
        rows = [r for r in prior['pool_rows'] if r['subset'] != 'bridge_positive'] + members
        draws = {}; swaps = {}; datasets = {}; ledger = {}
        for seed in SEEDS:
            a,b,s = schedules(prior,members,seed); swaps[str(seed)] = s
            for arm,seq in (('K',a),('L',b)):
                key = f'{arm}-300-{seed}'; draws[key] = seq; ledger[key] = exposures(rows,seq)
                listing = run/f'{key}.txt'; lookup = {r['member_id']:r for r in rows}
                listing.write_text('\n'.join(lookup[x]['image_path'] for x in sorted(set(seq)))+'\n')
                config = run/f'{key}.yaml'
                config.write_text(f'path: {run}\ntrain: {listing}\nval: {listing}\nnames: [transformer, switchgear, capacitor_bank, reactor]\n')
                datasets[key] = str(config)
                for p in (listing,config): inputs[str(p)] = file_sha256(p)
        for r in rows:
            for kind in ('image','label'):
                if file_sha256(r[kind+'_path']) != r[kind+'_sha256']: raise ValueError('Stale training member')
                inputs[r[kind+'_path']] = r[kind+'_sha256']
        inputs[str(dataset)] = file_sha256(dataset)
        save(target,dict(status='frozen_design_execution_preflight_required',dataset_path=str(dataset),pool_rows=rows,
            schedules=draws,exposures=ledger,paired_swaps=swaps,datasets=datasets,seeds=list(SEEDS),steps=300,
            controls={**prior['controls'],'optimizer_steps':300,'device':'cpu','warmup_epochs':0,'learning_rate':.001},
            acceptance_policy=prior['acceptance_policy'],retention=prior['retention'],
            candidate_priority=['L-300'],reference_family='K-300',historical_references=['R-300','historical_A'],
            additional_retention_reference='K-300',formal_confidence=.37,diagnostic_confidence=.001,
            evaluation=dict(input_size=640,device='cpu',class_aware_nms=True,nms_iou=.7,max_det=300,matching_iou=.5,
                paired_development_frames=48,no_target_development_frames=48,unseen_scene='sealed_not_evaluated'),
            quota_per_600=dict(base=216,regular=156,appearance_paired=120,hard_negative=108),
            interpretation='K uses new original views; L swaps only the same slots to matched appearance/light variants. Common members, positions, class-instance counts and pose exposure identical. Joint material/light intervention, not isolated material or light causal effects. Comparisons to I also change pose and supervision distribution.',
            launch_gate='Not a runnable completion or admission. Before launch: validate runner enforces lr=.001, exact schedules and all six cells; bind runner/test hashes in an independent execution receipt. Preserve failed attempts; no sealed tests.',
            limitations=['56 images share one map/assets; 8 poses, 7 source groups; no independent-scene claim.',
                'Switchgear front panels remain uncovered. Existing base lineage may be member-only; do not infer full source isolation.',
                'AI visual review is not pixel instance certification. Final validation on training members is training-fit only.'],
            baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),inputs=inputs,
            training_started=False,max_attempts=3))
        print('FROZEN', target, '56 images; 77 boxes; 6 schedules; no training', flush=True)
    except BaseException:
        save(run/'failure.json',dict(status='failed',error=traceback.format_exc()))
        raise

if __name__ == '__main__': main()
