"""Explicit observations of the reflection arm, not automatic label approval."""
from pathlib import Path
from scripts.vision import record_routed_gray_results_review as writer
from scripts.vision.routed_reflection_control import OUT as TRAIN, checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES = {
0:{1:'青灰块体大正面与完整基座清楚，斜线为主体上的阴影，无主要前景遮挡。'},
1:{1:'后变压器三柱、顶面与主体上部可见，下部被近变压器遮挡，前景柱不属于它。',2:'近变压器三柱、顶面及两大主体面清楚，下图缘截断。',3:'后电容器顶面及上部可见，下部被青色块体与近变压器遮挡。'},
2:{2:'左缘柜体大侧背面、顶面与基座可见，左图缘截断，无正面面板。',3:'后圆柱顶部和主体上部可见，下部被变压器遮挡，前景小柱不是圆柱部件。'},
3:{7:'右缘圆柱顶面、主体与基座部分可见，右图缘截断。'},
4:{1:'独立圆柱主体与完整基座清楚，无主要前景遮挡。'},
5:{4:'右柜面板、顶面、侧面及基座清楚，左下小角被前柜遮挡。'},
6:{4:'近变压器大顶面、主体及两柱可见，右下图缘截断。'},
7:{0:'后柜顶面、右侧及面板部分可见，左部被近柜、下部被变压器遮挡；前景柱不属于柜体。',3:'近柜顶面、面板、大主体面及基座清楚。',4:'近变压器大顶面、主体及两柱可见，右下图缘截断。'},
8:{0:'后柜顶部与窄带可见，下部被同类队列遮挡，框内含前柜。',4:'右柜面板、顶面、侧面与基座清楚，左下角被近柜遮挡。'},
9:{2:'队列后柜顶面及窄带可见，下部被近柜遮挡。',8:'远柜右侧、顶面与基座可见，左部被变压器遮挡，尺度较小。'},
10:{1:'青灰块体大主体面和完整基座清楚，无主要前景遮挡，主体上有斜向阴影。'},
11:{0:'左近柜体大顶面、侧面和基座局部可见，左下图缘截断。',1:'独立圆柱主体与完整基座清楚，未见主要遮挡。'},
12:{2:'后柜顶面及窄带可见，下部被同类近柜遮挡。',4:'变压器三柱、顶面、大主体面与完整基座清楚。',9:'后电容器顶面及主体上部可见，下部被变压器遮挡，框内三柱属于前景。'},
13:{7:'右缘圆柱顶面、主体和基座部分可见，右图缘截断。'},
14:{1:'右图缘柜体侧面与基座局部可见，右缘截断且前柜遮挡。'},
15:{3:'右侧队列后柜顶部及窄带可见，下部被前柜遮挡，框包含多个前景柜体。'},
}
MATERIAL = {
0:{1:'后灰柜顶面、大侧面及面板部分可见，左前被近柜遮挡。',2:'灰柜顶面、侧面、面板轮廓与基座清楚。'},
1:{2:'左缘灰柜大侧背面、顶面及基座清楚，左图缘截断，未见正面面板。'},
2:{4:'灰柜面板轮廓、顶面、侧面与右基座清楚，左下被前柜遮挡。'},
3:{0:'右灰变压器顶柱、顶面、大主体面及基座可见，右图缘截断。'},
4:{1:'灰圆柱主体及完整基座清楚，无主要前景遮挡，不是仅基座片段。'},
5:{4:'左灰变压器顶柱、顶面、主体及基座可见，粗杆遮挡且左图缘截断。',6:'中灰柜顶面、主体上部和基座部分可见，下部被近变压器遮挡。'},
6:{3:'最远队列灰柜顶面与窄带可见，下部被近柜遮挡，框包含前景柜体。',1:'近队列灰柜大顶面与窄主体带可见，下部被更近柜遮挡。'},
7:{2:'后灰电容器顶面、大主体面与右基座可见，左下被青色块体遮挡。'},
}
FP = {
0:('mixed_structure','右竖杆下段、杆脚及地面网格阴影，不是青色块体。'),
1:('gray_block_body','灰块体大侧背面、右暗面板与基座，主要杆体和青色块体在框外。'),
2:('gray_block_body','冷暗灰块体侧背面、右暗面板与基座，右缘含细杆和少量青色局部。'),
3:('gray_block_body','灰块体正面暗面板、灰边框及基座。'),
4:('gray_block_body','冷暗灰块体正面暗面板、灰边框及基座，未包含主要杆体。'),
5:('gray_block_body','灰块体正面暗面板、灰边框及基座，与同图其他seed为同一结构。'),
6:('gray_block_body','灰块体大侧背面、右暗面板与基座，与同图其他seed为同一结构。'),
7:('gray_block_body','冷暗灰块体大侧背面、右暗面板及基座，右缘含少量杆体及青色局部。'),
8:('mixed_structure','灰块体侧背面、穿过主体的前景粗杆及右下青色块体共同入框。'),
9:('mixed_structure','冷暗灰块体侧背面、前景粗杆及右下青色块体共同入框。'),
10:('gray_block_body','冷暗灰块体正面暗面板、灰边框与基座，与同图其他seed为同一结构。'),
11:('gray_block_body','右图缘严重截断的灰块体暗面板、边框与基座，左侧含围墙地面，不是中间青色块体。'),
12:('gray_block_body','冷暗右图缘灰块体面板、边框与基座局部，严重截断，不是中间青色块体。'),
13:('gray_block_body','灰块体两大主体面与基座，右边界含少量前景杆体。'),
14:('mixed_structure','灰块体两大主体面、基座及右侧前景粗杆共同入框。'),
}

def run():
    # Reuse the validated serializer only; observations above are current-arm evidence.
    writer.TRAIN=TRAIN;writer.OUT=TRAIN/'audit-v1'
    writer.NOTES=NOTES;writer.MATERIAL=MATERIAL;writer.FP=FP
    dest=writer.OUT/'review.json'
    result=checked(dest) if dest.exists() else writer.run()
    bound=writer.OUT/'review-author-receipt.json'
    paths=[dest,Path(__file__),Path(writer.__file__)]
    if bound.exists(): return checked(bound)
    return write_record(bound,dict(status='explicit_current_reflection_observations_bound',
        counts=result['counts'],training_admitted=False,promotable=False,
        inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__': print(run()['counts'])
