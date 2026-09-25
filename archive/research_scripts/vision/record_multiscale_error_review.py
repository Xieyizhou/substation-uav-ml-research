"""Explicit observations from the 27 full-frame/crop cards; not auto approval."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.diagnose_multiscale_errors import DEST,prior

OBS={
 'E01':['青色柜状块体侧背面、顶面和基座，同一结构被预测两次。','同一青色块体顶面与侧背面局部，较小框不包含完整基座。'],
 'E02':['青色柜状块体正面深色面板、侧面及基座。','后方灰色块体下部及基座横条，左侧含前景青色块体，不是完整变压器。'],
 'E03':['青色柜状块体正面面板、侧面与基座可辨；误检框不覆盖旁边杆体。'],
 'E04':['左图缘青色柜状块体面板、顶面、右侧及基座，外形部分出画。'],
 'E05':['冷暗光下左缘青色柜状块体面板与侧面。','后方灰色矩形块体正面深色面板和基座，不能由外形推断资产类别。'],
 'E06':['柜体正面深色面板、顶面和基座可辨；左下被前景柜体遮挡。'],
 'E07':['近景变压器宽主体、顶面三个端子和基座清楚；不是微小目标。'],
 'E08':['右图缘多柜紧邻、重叠且出画；可见窄侧面与顶面，目标像素分界无法可靠确认。'],
 'E09':['孤立柜体宽侧背面、顶面与基座清楚，面板背向相机。'],
 'E10':['近景柜体宽顶面、侧背面可见，右侧与下部出画。'],
 'E11':['后方电容器组顶面、宽侧面及基座可辨；左下局部被前景柜体遮挡，没有可见变压器端子。'],
 'E12':['柜体正面面板和顶面可辨，左下被前景柜体遮挡。'],
 'E13':['多柜前后叠置，仅顶面及横向侧面条带；框内目标像素归属无法可靠确认。'],
 'E14':['右缘变压器宽主体、顶面与端子可辨，右侧出画。'],
 'E15':['近景变压器宽顶面及三个端子可见，下部与右侧截断；内容丰富。'],
 'E16':['近景变压器宽顶面、侧面与三个端子清楚，下部出画。'],
 'E17':['原始条件右缘邻柜重叠，顶面与窄侧面可见，实例像素分界无法可靠确认。'],
 'E18':['中景柜体完整宽背侧面、顶面与基座可辨，无明显遮挡。'],
 'E19':['左缘近景变压器宽顶面与多个端子清楚，左侧和下部出画。'],
 'E20':['灰色电抗器圆顶与上部圆柱可辨，下部被前景变压器遮挡；框内端子属于前景，不归给电抗器。'],
 'E21':['变压器主体、三个端子与基座完整可辨，无显著遮挡。'],
 'E22':['左缘柜体宽侧背面与基座可见，部分外形出画，无正面面板。'],
 'E23':['右缘变压器主体与三个端子清楚，右側出画。'],
 'E24':['灰色电抗器圆柱主体和基座清楚，未见明显前景遮挡。'],
 'E25':['后方电容器组顶面与上部侧面可见，下部被前景柜体和变压器遮挡；前景端子不能算目标内容。'],
 'E26':['右缘电抗器圆柱局部、顶面和基座可辨，右側截断。'],
 'E27':['灰色电抗器圆柱局部与基座可辨，右缘截断，没有可见面板。'],
}
PENDING={'E08','E13','E17'}

def validate(e,ds):
    expected={f"{x['event_id']}:{j}":x for x in e['events'] for j in range(len(x['events']) if x['kind']=='FP' else 1)}
    if len(ds)!=len(expected) or {d['decision_id'] for d in ds}!=set(expected):raise ValueError('Missing/duplicate decision')
    for d in ds:
        x=expected[d['decision_id']]
        if d['image_sha256']!=x['source']['image_sha256'] or d['evidence_sha256']!=x['page_sha256']:raise ValueError('Stale decision')
        if prior.file_sha256(x['page_path'])!=d['evidence_sha256']:raise ValueError('Changed page')
        if not d['reason'] or d['review_nature']!='AI-assisted' or not d['reviewed_at']:raise ValueError('Invalid decision')

def main():
    ep=DEST/'evidence.json';e=prior.read(ep);prior.verify(e);ds=[]
    for x in e['events']:
        notes=OBS[x['event_id']]
        if len(notes)!=(len(x['events']) if x['kind']=='FP' else 1):raise ValueError('Explicit observations mismatch')
        for j,note in enumerate(notes):
            ds.append(dict(decision_id=f"{x['event_id']}:{j}",status='pending' if x['event_id'] in PENDING else 'reviewed',
                reason=note,review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
                image_sha256=x['source']['image_sha256'],evidence_sha256=x['page_sha256'],
                prediction=x['events'][j] if x['kind']=='FP' else None,truth=x.get('truth'),
                pixel_visibility_certified=False,asset_identity_inferred_from_appearance=False))
    validate(e,ds)
    prior.frozen(DEST/'review.json',dict(status='review_complete_with_named_gaps',decisions=ds,pending_ids=sorted(PENDING),
        selected_candidate=None,inputs={str(p):prior.file_sha256(p) for p in (ep,Path(__file__).resolve())}))
    print('DECISIONS',len(ds),'PENDING',sorted(PENDING))

if __name__=='__main__':main()
