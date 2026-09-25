"""Read-only Q analysis; new hash-bound artifacts, no implicit visual approval."""
from collections import Counter
from pathlib import Path
import csv
from PIL import Image, ImageDraw
from scripts.vision.run_visibility_quality_training import OUT as RUN, read, save, file_sha256, verify_tree, SEEDS, HIST_TRAIN, BASE, trainer
from scripts.vision.verify_experiment_baseline import verify as verify_baseline
from scripts.vision.exposure_protocol import ROOT

OUT = RUN / 'error-diagnosis-v1'

def paired_change(before, after):
    a = {(r['pair_id'], r['variant']): r for r in before}
    b = {(r['pair_id'], r['variant']): r for r in after}
    if len(a) != len(before) or a.keys() != b.keys():
        raise ValueError('Incomplete or duplicate pair identities')
    result = Counter()
    for key, row in a.items():
        other = b[key]
        if row['truth'] != other['truth'] or row['image_sha256'] != other['image_sha256']:
            raise ValueError('Changed paired evidence')
        x, y = row['planned_assigned_hit'], other['planned_assigned_hit']
        result[key[1] + ':' + ('gain' if y and not x else 'loss' if x and not y else 'retained_hit' if y else 'retained_miss')] += 1
    return dict(result)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'analysis.json').exists():
        verify_tree(OUT/'analysis.json'); print('VERIFIED_EXISTING', OUT); return
    verify_tree(RUN/'completion.json')
    p = read(RUN/'protocol.json'); hist = read(HIST_TRAIN/'protocol.json')
    inputs = {str(x): file_sha256(x) for x in (RUN/'completion.json', HIST_TRAIN/'protocol.json', Path(__file__))}
    cells = {}; paired = {}; prefix = {}; exposure = {}; items = []
    sources = {(r['view_id'], r['variant']): r for r in read(BASE/'hard-negative-isolated-v2/semantic-review.json')['frames']}
    for seed in SEEDS:
        records = {}; losses = {}
        for steps in (100, 300):
            key = f'Q-{steps}-{seed}'; cp = RUN/key/'completion.json'
            cell = trainer.checked_cell(cp, p)
            path = RUN/f'evaluation-{key}.json'; r = read(path); records[steps] = r
            inputs[str(cp)] = file_sha256(cp); inputs[str(path)] = file_sha256(path)
            loss_path = Path(cell['exposure_path']).parent/'results.csv'
            with loss_path.open() as stream:
                losses[steps] = [{k.strip():v.strip() for k,v in row.items() if 'train/' in k} for row in csv.DictReader(stream)]
            reasons = Counter(); planned_reasons = Counter()
            for row in r['rows']:
                if row['matching_conflict']: raise ValueError('Unresolved matching conflict')
                for miss in row['misses']:
                    name = ':'.join((row['variant'], miss['class_name'], miss['reason']))
                    reasons[name] += 1
                    if miss['truth_index'] == row['planned_truth_index']: planned_reasons[name] += 1
            cells[key] = dict(summary=r['summary'], negative_summary=r['negative_summary'], miss_events=dict(reasons), planned_miss_events=dict(planned_reasons))
            if steps == 300:
                for row in r['negative_rows']:
                    for index, pred in enumerate(row['predictions']):
                        src = sources[row['view_id'], row['variant']]
                        if file_sha256(src['image_path']) != row['image_sha256']: raise ValueError('Stale negative image')
                        inputs[src['image_path']] = row['image_sha256']
                        items.append(dict(item_id=f'F{len(items)+1:02}', seed=seed, prediction_index=index, view_id=row['view_id'], image_path=src['image_path'], image_sha256=row['image_sha256'], prediction=pred, review_status='pending'))
        prefix[str(seed)] = dict(schedule=p['schedules'][f'Q-300-{seed}'][:600] == p['schedules'][f'Q-100-{seed}'], train_loss=losses[300][:len(losses[100])] == losses[100])
        if not all(prefix[str(seed)].values()): raise ValueError('Training prefix mismatch')
        paired[str(seed)] = paired_change(records[100]['rows'], records[300]['rows'])
        exposure[str(seed)] = dict(before=hist['exposures'][f'N-100-{seed}']['class_instance_exposure'], after=p['exposures'][f'Q-100-{seed}']['class_instance_exposure'])
    items.sort(key=lambda x:(x['image_sha256'], x['seed'], x['prediction_index']))
    pages = []
    for offset in range(0, len(items), 6):
        canvas = Image.new('RGB', (1200, 900), 'white'); draw = ImageDraw.Draw(canvas)
        for n, item in enumerate(items[offset:offset+6]):
            im = Image.open(item['image_path']).convert('RGB'); box = item['prediction']['bbox_xyxy']
            crop = im.crop((max(0,int(box[0])),max(0,int(box[1])),min(im.width,int(box[2])+1),min(im.height,int(box[3])+1)))
            crop.thumbnail((580,235)); x=n%2*600; y=n//2*300
            canvas.paste(crop,(x,y+60))
            draw.text((x+5,y+5),f"{item['item_id']} seed={item['seed']} {item['prediction']['class_name']}\nconf={item['prediction']['confidence']:.4f} image={item['image_sha256'][:12]}",fill='black')
        page=OUT/f'negative-page-{offset//6+1:02}.png'; canvas.save(page); pages.append(str(page)); inputs[str(page)]=file_sha256(page)
        for item in items[offset:offset+6]: item['evidence_path']=str(page); item['evidence_sha256']=inputs[str(page)]
    result=save(OUT/'analysis.json', dict(status='quantitative_analysis_complete_visual_review_pending', cells=cells, paired_budget_change=paired, prefix_checks=prefix, exposure_change=exposure, negative_review_items=items, negative_unique_images=len({x['image_sha256'] for x in items}), pages=pages, baseline=verify_baseline(ROOT/'config/perception/visual_experiment_baseline_v1.json'), inputs=inputs, interpretation='Repeated seeds and paired variants are prediction events, not independent scenes. Member-only lineage IDs do not establish scene independence. No new training, label revision or admission in this analysis.'))
    print('ANALYSIS',OUT,'FP_EVENTS',len(items),'UNIQUE_IMAGES',result['negative_unique_images'],'PREFIX',prefix,flush=True)

if __name__ == '__main__': main()
