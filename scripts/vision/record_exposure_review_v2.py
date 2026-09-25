"""Explicit re-review of all 30 correctly separated light-variant images."""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import OUT, read, save, verify, file_sha256
from scripts.vision.analyze_recovery_paired_calibration import iou

# Individually read from sheets review-evidence-v2/00.png through 29.png.
# B building, P pole, G ground/shadow, C cabinet, M mixed structures.
OBSERVATIONS = {
    0: ('BPB', '建筑背面及底座；中间编号1主要覆盖右侧杆体上半部和横杆'),
    1: ('G', '底缘细长框覆盖地面网格'),
    2: ('B', '低冷光下右缘建筑前面板、侧面与底座'),
    3: ('BBGB', '正常光下编号0、1、3覆盖建筑；编号2覆盖地面和底座局部'),
    4: ('B', '低冷光建筑前面、右侧立面和上方小段杆体'),
    5: ('BBB', '正常光下三个框均覆盖同一建筑前面和侧面'),
    6: ('C', '蓝绿色普通柜体背面及黑色底座'),
    7: ('G', '底缘网格与杆体投影'),
    8: ('BBBGB', '编号0、1、2、4覆盖右缘建筑；编号3为底缘网格和建筑底座边缘'),
    9: ('BBBGBG', '编号0、1、2、4为建筑背面；编号3、5为地面网格细长框'),
    10: ('M', '低冷光下建筑前面、前方杆体和柜体顶部共同入框'),
    11: ('CG', '正常光编号0为蓝绿色柜体，编号1为底缘地面和杆影'),
    12: ('M', '宽框含大面积地面、杆体下部和远处边界'),
    13: ('GG', '正常光下两个细长框均为画面底缘网格'),
    14: ('BB', '低冷光下两个框均覆盖左侧建筑及邻近地面'),
    15: ('BBGB', '正常光编号0、1、3覆盖左侧建筑；编号2为底缘网格和杆影'),
    16: ('B', '低冷光下右缘截断建筑前面板'),
    17: ('CBB', '正常光编号0为蓝绿色柜体；编号1、2为右缘建筑前面板'),
    18: ('M', '低冷光下后方建筑、遮挡杆体和前景柜体顶部共同入框'),
    19: ('C', '正常光下前景柜体前面板及底座'),
    20: ('MMG', '编号0、1为远处建筑、杆体与柜体重叠结构；编号2为底缘地面网格'),
    21: ('P', '左缘截断杆体下部及少量地面'),
    22: ('B', '低冷光下远处灰色建筑前面板和底座'),
    23: ('BBGB', '正常光编号0、1、3为远处建筑，编号2为底缘地面和阴影'),
    24: ('B', '低冷光下灰色建筑背面、侧面及下方地面'),
    25: ('BBB', '正常光下三个框均覆盖同一建筑背面和侧面'),
    26: ('BBB', '三个框均覆盖右缘截断建筑、底座及邻近地面'),
    27: ('M', '低冷光大框同时包含建筑、多根杆体、柜体和地面'),
    28: ('BB', '正常光下两个框主要覆盖建筑背面，右边缘含遮挡杆体'),
    29: ('BBBM', '编号0、1、2主要覆盖左缘建筑背面；编号3横跨建筑底座、地面阴影和柜体下部'),
}
CONTENTS = {'B': 'building', 'P': 'pole', 'G': 'ground_shadow', 'C': 'cabinet', 'M': 'mixed_structure'}


def main():
    manifest_path = OUT/'review-manifest-v2.json'
    manifest = read(manifest_path)
    verify(manifest)
    inputs = {str(manifest_path): file_sha256(manifest_path), str(Path(__file__)): file_sha256(Path(__file__))}
    rows = []
    for frame in manifest['frames']:
        choices, reason = OBSERVATIONS[frame['number']]
        if len(choices) != len(frame['predictions']):
            raise ValueError('Explicit review count mismatch')
        for kind in ('image', 'evidence'):
            if file_sha256(frame[f'{kind}_path']) != frame[f'{kind}_sha256']:
                raise ValueError('Re-reviewed image changed')
            inputs[frame[f'{kind}_path']] = frame[f'{kind}_sha256']
        for index, (prediction, choice) in enumerate(zip(frame['predictions'], choices)):
            rows.append({**prediction, 'view_id': frame['view_id'], 'variant': frame['variant'],
                'image_sha256': frame['image_sha256'], 'evidence_sha256': frame['evidence_sha256'],
                'content_category': CONTENTS[choice], 'decision': 'reviewed',
                'reason': f'已核对该光照原图及编号{index}裁剪；{reason}', 'review_nature': 'AI-assisted',
                'reviewer': 'Codex visual re-inspection', 'reviewed_at': datetime.now(timezone.utc).isoformat(),
                'overlapping_box_seeds_iou_ge_05': sorted({p['seed'] for p in frame['predictions'] if iou(p['bbox_xyxy'], prediction['bbox_xyxy']) >= .5})})
    if len(rows) != 67 or len({r['prediction_id'] for r in rows}) != 67:
        raise ValueError('Prediction identity collision')
    result = save(OUT/'visual-review-v2.json', {'status': 'reviewed', 'decisions': rows, 'unique_images': 30,
        'unique_pose_ids': 20, 'prediction_count': 67, 'inputs': inputs,
        'supersedes': 'visual-review.json; previous view-only indexing merged lighting variants'})
    save(OUT/'review-index-erratum.json', {'status': 'corrected_and_re_reviewed',
        'reason': 'view_id identifies pose, not image; variant must participate in lookup and prediction identity.',
        'superseded_artifacts': ['review-manifest.json', 'visual-review.json', 'review-evidence/'],
        'active_artifacts': ['review-manifest-v2.json', 'visual-review-v2.json', 'review-evidence-v2/'],
        'training_or_inference_changed': False, 'correct_unique_images': 30, 'unique_poses': 20,
        'inputs': {str(OUT/'visual-review-v2.json'): file_sha256(OUT/'visual-review-v2.json')}})
    print(result['identity'])


if __name__ == '__main__':
    main()
