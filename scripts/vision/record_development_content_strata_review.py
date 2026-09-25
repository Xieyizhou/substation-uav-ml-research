"""Record explicit, prediction-free content strata decisions for all 120 truths."""
from datetime import datetime, timezone
import re
from pathlib import Path
from scripts.vision.evaluate_unified_lighting import OUT as RUN, prior
from scripts.vision.prepare_development_content_strata import OUT, RULES

# Pair/target decisions are based only on the frozen no-prediction pages.
STRATA = {
    1: ['clear', 'partial'],
    2: ['partial', 'clear'],
    3: ['clear', 'clear', 'partial', 'fragment', 'fragment', 'fragment'],
    4: ['partial', 'clear', 'partial', 'partial'],
    5: ['partial', 'partial', 'clear', 'clear', 'partial', 'partial'],
    6: ['partial', 'partial', 'partial', 'partial', 'fragment', 'clear', 'partial', 'partial', 'fragment'],
    7: ['partial', 'partial', 'partial', 'clear', 'fragment', 'fragment', 'fragment', 'fragment', 'fragment', 'fragment', 'clear'],
    8: ['fragment', 'clear'],
    9: ['partial', 'clear', 'clear', 'clear', 'fragment', 'fragment', 'partial', 'partial', 'partial', 'partial'],
    10: ['clear'],
    11: ['partial', 'partial', 'partial'],
    12: ['clear', 'clear', 'fragment', 'partial'],
}

REASONS = {
    'clear': '主体外形和较丰富表面可辨，未见显著前景遮挡；虽不要求正面面板，但目标可独立分离。',
    'partial': '目标主体仍可分离并有较丰富外形，但受前景遮挡、杆体或图缘影响，完整外形不可见。',
    'fragment': '仅见顶沿、窄侧面、图缘片段或被邻物覆盖的局部，内容不足以稳定辨识完整目标。',
    'unknown': '无法可靠判断目标与邻物的像素归属或可辨内容。',
}

def instance_id(t):
    return int(re.search(r'-instance-(\d+)-', t['annotation_id']).group(1))

def main():
    pp = RUN / 'protocol.json'
    protocol = prior.read(OUT / 'protocol.json')
    prior.verify(protocol)
    if set(STRATA) != {x['pair_number'] for x in protocol['records']}:
        raise ValueError('Pair coverage changed')
    now = datetime.now(timezone.utc).isoformat()
    decisions = []
    for x in protocol['records']:
        pair, target = x['pair_number'], x['target_number']
        choices = STRATA[pair]
        if target > len(choices): raise ValueError('Target coverage changed')
        s = choices[target - 1]
        source = x['source']
        page = Path(x['page'])
        if not page.exists() or prior.file_sha256(source['image_path']) != source['image_sha256']:
            raise ValueError('Stale image or page')
        decisions.append(dict(
            decision_id=x['review_id'], pair_number=pair, target_number=target,
            pair_id=source['pair_id'], view_id=source['view_id'], variant=source['variant'],
            instance_id=instance_id(x['truth']), object_id=source['expected_object_id'],
            class_name=x['truth']['class_name'], truth=x['truth'], truth_sha256=source['truth_sha256'],
            stratum=s, reason=REASONS[s], review_nature='AI-assisted',
            reviewed_at=now, image_path=source['image_path'], image_sha256=source['image_sha256'],
            page= str(page), page_sha256=prior.file_sha256(page), crop_path=source['crop_path'],
            crop_sha256=source['crop_sha256'], training_admitted=False, promotable=False,
        ))
    ids = [x['decision_id'] for x in decisions]
    if len(decisions) != 120 or len(set(ids)) != 120:
        raise ValueError('Expected 120 unique decisions')
    # A source edit invalidates the previous frozen attempt; preserve it and write a new identity.
    prior.frozen(OUT / 'review-v2.json', dict(status='content_strata_review_complete', decisions=decisions,
        rules=RULES, distribution={s: sum(x['stratum'] == s for x in decisions) for s in RULES},
        scope='Prediction-free visual review; already-viewed development data; no label correction, exclusion, training admission or promotion.',
        inputs={str(x): prior.file_sha256(x) for x in (OUT/'protocol.json', pp, Path(__file__).resolve())}))
    print('CONTENT_STRATA_REVIEW_COMPLETE', len(decisions), {s: sum(x['stratum'] == s for x in decisions) for s in RULES})

if __name__ == '__main__': main()
