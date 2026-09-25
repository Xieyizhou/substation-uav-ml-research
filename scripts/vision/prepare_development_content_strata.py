"""Freeze all original/lighting truths; render prediction-free review pages."""
from pathlib import Path
import re
from PIL import Image, ImageDraw
from scripts.vision.evaluate_unified_lighting import OUT as RUN, prior, paired_truth

OUT = RUN.parents[1] / 'development-content-strata-v1'
RULES = {
    'clear': '主体外形及较丰富表面内容可辨，未见显著前景遮挡或图缘截断；不要求正面可见。',
    'partial': '有局部遮挡或边界影响，但仍能分离目标并观察较丰富主体外形；不是仅窄条。',
    'fragment': '只见顶沿、窄侧面、图缘片段或其他不足内容；实例存在不等于类别可辨。',
    'unknown': '无法可靠判断目标与邻物归属或可辨内容；不能自动排除。',
}

def instance_key(t):
    return int(re.search(r'-instance-(\d+)-', t['annotation_id']).group(1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pp = RUN / 'protocol.json'
    rp = Path(prior.read(pp)['evaluation']['paired_review'])
    review = prior.read(rp)
    prior.verify(review)
    pairs, inputs = paired_truth([x for x in review['frames'] if x['variant'] in ('original', 'lighting')])
    inputs.update({str(x): prior.file_sha256(x) for x in (pp, rp, Path(__file__).resolve())})
    ordered = sorted({r['view_id'] for r, _ in pairs})
    records = []
    for n, view in enumerate(ordered, 1):
        frames = {r['variant']: (r, ts) for r, ts in pairs if r['view_id'] == view}
        if set(frames) != {'original', 'lighting'}: raise ValueError('Missing pair')
        a, b = [frames[v][1] for v in ('original', 'lighting')]
        aa, bb = ({instance_key(t): t for t in group} for group in (a, b))
        if len(aa)!=len(a) or len(bb)!=len(b) or aa.keys()!=bb.keys(): raise ValueError('Instance mapping differs')
        for k in aa:
            if {f:v for f,v in aa[k].items() if f!='annotation_id'} != {f:v for f,v in bb[k].items() if f!='annotation_id'}: raise ValueError('Paired truth differs')
        for variant in ('original', 'lighting'):
            row, _ = frames[variant]
            ts = sorted(frames[variant][1], key=instance_key)
            if prior.file_sha256(row['image_path']) != row['image_sha256']: raise ValueError('RGB changed')
            inputs[row['image_path']] = row['image_sha256']
            for j, truth in enumerate(ts, 1):
                records.append(dict(review_id=f'P{n:02}-{j:02}-{variant}', pair_number=n, target_number=j,
                    source=row, truth=truth, page=str(OUT / f'P{n:02}.png')))
    if len(pairs) != 24 or len(records) != 120: raise ValueError('Unexpected frozen scope')
    prior.frozen(OUT / 'protocol.json', dict(status='rules_and_all_truths_frozen_before_stratified_scores',
        rules=RULES, precedence='unknown if attribution uncertain, otherwise fragment/partial/clear by observed content; no pixel-area threshold',
        review_caveat='Already-viewed development data, not blind review; pages omit predictions. No outcome-based label removal or new pass threshold.',
        records=records, inputs=inputs))
    paths = [OUT / 'protocol.json', Path(__file__).resolve()]
    for n, view in enumerate(ordered, 1):
        rows = [x for x in records if x['pair_number'] == n]
        m = max(x['target_number'] for x in rows)
        canvas = Image.new('RGB', (1600, 480 + 200*m), 'white')
        d = ImageDraw.Draw(canvas)
        for col, variant in enumerate(('original', 'lighting')):
            r = next(x['source'] for x in rows if x['source']['variant'] == variant)
            im = Image.open(r['image_path']).convert('RGB')
            full = im.copy(); fd = ImageDraw.Draw(full)
            for x in rows:
                if x['source']['variant'] != variant: continue
                box = x['truth']['bbox_xyxy']; j = x['target_number']
                fd.rectangle(box, outline='red', width=3); fd.text((box[0]+3, box[1]+3), str(j), fill='yellow', stroke_width=1, stroke_fill='black')
                crop = im.crop(box); crop.thumbnail((760, 170))
                canvas.paste(crop, (col*800, 500+(j-1)*200))
                d.text((col*800, 480+(j-1)*200), f'{j} {x["truth"]["class_name"]} {x["review_id"]}', fill='black')
            full.thumbnail((800, 450)); canvas.paste(full, (col*800, 25))
            d.text((col*800, 5), f'P{n:02} {variant}', fill='black')
        dest = OUT / f'P{n:02}.png'; canvas.save(dest); paths.append(dest)
    prior.frozen(OUT / 'evidence.json', dict(status='awaiting_explicit_all_truth_review',
        inputs={str(x): prior.file_sha256(x) for x in paths}))
    print(OUT, len(records), 'truths; 12 paired pages')


if __name__ == '__main__': main()
