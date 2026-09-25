"""Transcribe explicit Codex visual decisions made from the 20 evidence sheets."""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import OUT, read, save, verify, file_sha256
from scripts.vision.analyze_recovery_paired_calibration import iou

# Each entry was individually inspected in full-frame context and its box crop.
DECISIONS = {
    0: [('building', '灰色建筑背面及底座；上方含小段杆体'), ('pole', '框主要覆盖右侧杆体上半部与横杆'), ('building', '灰色建筑背面、底座和顶部露出的杆体')],
    1: [('ground_shadow', '画面底缘细长框，仅覆盖地面网格')],
    2: [('building', '右缘截断的灰色建筑前面板与底座'), ('building', '同一灰色建筑前面板'), ('ground_shadow', '底缘细长框含地面及杆脚、建筑底座的局部'), ('building', '灰色建筑前面板被异类重复框覆盖'), ('building', '较宽框覆盖同一建筑及少量周边天空地面')],
    3: [('building', '灰色建筑前面和右侧立面'), ('building', '同一建筑被预测为电抗器'), ('building', '建筑及上方少量杆体'), ('building', '同一建筑前面和侧面')],
    4: [('cabinet', '蓝绿色普通柜体背面及黑色底座')],
    5: [('ground_shadow', '画面底缘网格与杆体投影，框未覆盖设备主体')],
    6: [('building', '右缘灰色建筑前面板'), ('building', '同一建筑及屋顶边缘'), ('building', '同一建筑与较大天空区域'), ('ground_shadow', '底缘网格与建筑底座边缘'), ('building', '灰色建筑及上方天空')],
    7: [('building', '右侧灰色建筑背面及底座'), ('building', '同一建筑背面'), ('building', '建筑背面被预测为电抗器'), ('ground_shadow', '底缘网格细长框'), ('building', '同一建筑背面'), ('ground_shadow', '更宽的底缘地面网格与投影')],
    8: [('cabinet', '右侧蓝绿色柜体及前面板'), ('ground_shadow', '底缘网格与杆脚投影'), ('mixed_structure', '建筑前面、前方杆体及柜体顶部共同入框')],
    9: [('ground_shadow', '右下底缘地面网格'), ('ground_shadow', '左下底缘地面网格'), ('mixed_structure', '宽框覆盖地面、杆体下部及远处边界')],
    10: [('building', '左侧灰色建筑及前面板'), ('building', '同一建筑被预测为电容器组'), ('ground_shadow', '底缘地面及杆影细长框'), ('building', '灰色建筑本体'), ('building', '同一建筑被重复预测为电容器组'), ('building', '较宽框覆盖建筑及邻近地面')],
    11: [('cabinet', '蓝绿色普通柜体前面板与底座'), ('building', '右缘灰色建筑前面板'), ('building', '同一建筑前面板'), ('building', '同一建筑及底座')],
    12: [('cabinet', '前景蓝绿色柜体及前面板'), ('mixed_structure', '后方建筑被杆体遮挡，框还包含前景柜体顶部')],
    13: [('mixed_structure', '远处灰色建筑与前景杆体重叠，含柜体一角'), ('mixed_structure', '建筑、遮挡杆体及柜体共同入框'), ('ground_shadow', '底缘地面网格细长框')],
    14: [('pole', '画面左缘截断的杆体下部及少量地面')],
    15: [('building', '远处灰色建筑前面板'), ('building', '同一建筑被预测为电抗器'), ('ground_shadow', '右下底缘网格及阴影'), ('building', '同一建筑被预测为电容器组'), ('building', '同一建筑前面及底座')],
    16: [('building', '左侧灰色建筑背面和侧面'), ('building', '同一建筑背面和侧面'), ('building', '同一建筑背面和侧面'), ('building', '同一建筑及下方地面')],
    17: [('building', '右缘截断的灰色建筑、底座及邻近地面'), ('building', '同一右缘建筑'), ('building', '同一右缘建筑')],
    18: [('building', '灰色建筑背面，右边缘含遮挡杆体'), ('mixed_structure', '大框同时覆盖建筑、多根杆体、柜体和地面'), ('building', '灰色建筑背面及边缘杆体')],
    19: [('building', '左缘灰色建筑背面及底座'), ('building', '同一建筑被预测为电抗器'), ('building', '同一建筑与较多天空'), ('mixed_structure', '底缘横框包含建筑底座、地面阴影和柜体下部')],
}


def main():
    manifest_path = OUT / 'review-manifest.json'
    manifest = read(manifest_path)
    verify(manifest)
    decisions = []
    inputs = {str(manifest_path): file_sha256(manifest_path), str(Path(__file__)): file_sha256(Path(__file__))}
    for frame in manifest['frames']:
        explicit = DECISIONS[frame['number']]
        if len(explicit) != len(frame['predictions']):
            raise ValueError('Visual decision count mismatch')
        for key in ('image', 'evidence'):
            if file_sha256(frame[f'{key}_path']) != frame[f'{key}_sha256']:
                raise ValueError('Reviewed evidence changed')
            inputs[frame[f'{key}_path']] = frame[f'{key}_sha256']
        for pred, (category, reason) in zip(frame['predictions'], explicit):
            overlapping = sorted({p['seed'] for p in frame['predictions'] if iou(p['bbox_xyxy'], pred['bbox_xyxy']) >= .5})
            decisions.append({**pred, 'view_id': frame['view_id'], 'image_sha256': frame['image_sha256'],
                'evidence_sha256': frame['evidence_sha256'], 'content_category': category,
                'reason': reason, 'decision': 'reviewed', 'review_nature': 'AI-assisted',
                'reviewer': 'Codex visual inspection', 'reviewed_at': datetime.now(timezone.utc).isoformat(),
                'overlapping_box_seeds_iou_ge_05': overlapping})
    result = save(OUT / 'visual-review.json', {'status': 'reviewed', 'unique_images': len(manifest['frames']),
        'prediction_count': len(decisions), 'decisions': decisions, 'inputs': inputs,
        'limits': ['Content categories describe pixels inside predictions, not independent scene counts.',
                   'Mixed structures are explicit visual categories, not uncertain class labels.']})
    print(result['identity'])


if __name__ == '__main__':
    main()
