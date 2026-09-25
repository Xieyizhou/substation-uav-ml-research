"""Explicit 155-box visual observations; no default pass decisions."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.risk_capped_control_review import OUT,ROOT,read,verify,frozen,file_sha256

# In label-line order. Written after viewing all 40 full images and 155 crops.
# Prefix ! means content insufficient/uncertain, not a claim of zero instance pixels.
OBS={
'C01':['完整块体、三柱附件及底座','!右图缘仅窄侧壁及底座片段，无完整主体'],
'C02':['后方主体上部及柱附件可见，下部被块体遮挡','圆柱主体可辨，前景杆遮住右侧','完整块体顶面侧面及底座'],
'C03':['后方块体与柱附件可辨，左部重叠','主体右段及两柱可辨，左部被圆柱遮挡','完整圆柱与底座'],
'C04':['图缘截断主体，仍有宽侧壁顶面及附件片段','完整主体三柱与底座','宽主体及柱附件，左侧截断','独立块体侧面顶边和底座'],
'C05':['独立块体完整顶面与宽侧面底座，非面板视角'],
'C06':['完整主体与三柱','完整面板、侧面及底座','块体宽主体可见，下角被前景柜体遮挡'],
'C07':['面板大部可见，左下被柜体挡住','完整面板和主体，左下小遮挡','!左下图缘仅顶面及侧壁，面板与完整轮廓不见','宽主体及柱附件，右图缘截断','完整面板与底座','独立圆柱及底座'],
'C08':['后方宽侧面部分被前景挡住，可辨块体但无面板','宽侧面和底座，左缘局部遮挡','宽主体与附件，杆体横穿','完整宽主体和底座、顶部附件','右图缘截断圆柱，曲面与底座仍可辨','!后方仅露出窄蓝色平面，前景遮住大部，归属内容不足'],
'C09':['上部主体及三柱可辨，下部被柜体遮挡','后方右段主体和附件可辨','右段主体和两柱可辨，左侧圆柱遮挡','完整柜状长侧面与底座，面板未见','完整圆柱主体及底座'],
'C10':['完整主体及三柱','主体三柱及底座，左图缘截断','完整长块体顶面侧面底座，非面板视角','侧壁顶面及底座大部可见，左角遮挡','完整独立块体和底座'],
'C11':['后排侧壁顶面及底座可见','!重叠块体仅上部平面，无法可靠区分完整目标','!多层重叠，目标可辨内容不足','!前景后只露出窄侧片，完整主体未见','!左图缘仅窄侧壁底座片段，附件未见','完整宽侧面顶边及底座，无面板视角','宽圆柱及底座，顶部图缘截断但曲面可辨'],
'C12':['主体、顶面和重叠柱附件可辨','完整长侧壁与底座，无面板视角','独立完整块体及底座'],
'C13':['完整主体、顶面三柱和底座','完整长侧面和底座','后方宽侧壁可辨，左部遮挡','完整块体和底座'],
'C14':['!左缘截断且前景叠压，仅上部侧片','顶面和三柱可辨，近距离图缘截断主体','!左图缘截断柜体，仅无面板侧壁及顶面','完整圆柱与底座'],
'C15':['宽主体和附件可辨，右图缘截断','完整圆柱及底座'],
'C16':['完整主体三柱及底座','!下图缘只剩顶面与柱附件，主体侧壁未见','目标宽侧壁顶面可辨，下角局部被柜体挡住'],
'C17':['主体三柱可辨，杆体挡住左边','长侧壁与底座可辨，杆遮右段','独立块体及底座完整'],
'C18':['近景大主体与三柱可辨，右边和底部截断','主体及附件可辨，中央被杆遮挡','宽主体底座可辨，右下角遮挡'],
'C19':['!灰色重叠块体只见上侧平面，面板和整体轮廓不足','灰色主体和底座可辨，杆遮中央，附件未见','!前景后方仅一段灰色上侧平面，结构归属不足','圆柱与底座可辨，左侧遮挡'],
'C20':['后方柜体面板轮廓可见，下部遮挡','灰色完整侧面顶面底座，左边图缘截断','完整主体三柱和底座','灰色完整面板、顶面侧面及底座','圆柱主体可辨，右下被前景挡住'],
'C21':['后方柜体面板部分可见，下部前景遮挡','宽面板与主体可辨，下缘截断','宽主体三柱和底座可辨，右缘截断','完整面板、主体及底座','圆柱主体可辨，下边被柜体遮住'],
'C22':['!前景后蓝色重叠上侧面，目标完整轮廓不足','主体侧面和底座可辨，杆遮中央','!前景间只见一段蓝色平面，实例内容不足','圆柱大部与底座可辨，左边前景遮挡'],
'C23':['宽主体三柱及底座可辨，右图缘截断','!左图缘柜体仅部分无面板侧壁，完整轮廓不足','圆柱与底座清晰'],
'C24':['完整俯视圆柱与底座'],
'C25':['后排侧壁顶面底座可辨，前景遮下角','后排宽侧面可辨，下边重叠','完整侧壁顶面底座','侧面底座大部可辨，左缘截断','!图缘极窄竖片，多为地面，目标主体不足','主体与三柱底座清晰','!前景变压器后仅上侧壁，缺少完整柜体内容','完整圆柱和底座'],
'C26':['后排面板大部可辨，两侧局部遮挡','!左图缘只见侧壁片段和底座，面板未见','主体三柱与底座可辨，右图缘截断','完整柜体面板与底座','完整圆柱与底座'],
'C27':['宽主体柱附件及底座可辨，右缘截断','宽顶面及上部侧壁和三柱可辨，近景截断','宽块体轮廓和底座可辨，左下遮挡'],
'C28':['完整主体附件及底座','宽主体及附件可辨，左图缘截断','完整宽侧壁与底座，下角局部遮挡'],
'C29':['主体及叠列附件可辨，底缘截断','后方宽侧壁仍可辨，下部局部被柜体遮挡'],
'C30':['!杆及前景后仅上侧片，多结构重叠','后排侧壁底座可辨，右下遮挡','后排侧壁底座可辨，杆遮左缘','!前景后仅窄侧片，完整柜体未见','主体三柱与底座可辨，左下遮挡','主体三柱可辨，下缘被块体遮挡','主体及附件可辨，中央杆遮挡','侧面底座部分可辨，右边遮挡','圆柱曲面和底座可辨，两边局部遮挡','!前景三柱后仅后排柜体上部，目标内容不足','完整俯视块体及底座'],
'C31':['完整圆柱与底座'],
'C32':['后排宽侧壁和底座可辨，下角遮挡','独立侧壁顶面及底座','!前景后仅窄侧片，杆及主体遮挡严重','主体三柱及底座可辨，左下遮挡','主体三柱可辨，下边被块体遮挡','!右图缘极窄片，多为地面与局部结构','右图缘侧壁和底座部分可辨','!前景后仅上部蓝色平面，完整柜体不足','宽顶面侧面可辨，底缘截断'],
'C33':['!下图缘仅顶面和单个柱附件，主体侧壁未见','完整主体三柱及底座','后排宽块体侧面底座可辨，右下遮挡'],
'C34':['!右图缘仅侧壁和底座片段，附件与完整主体未见','完整圆柱与底座'],
'C35':['完整块体顶面侧面底座'],
'C36':['后方上部主体三柱可辨，下部被柜体挡住','完整块体大部及底座，杆遮右缘'],
'C37':['!右图缘仅窄侧壁底座片段，无附件','完整圆柱与底座'],
'C38':['完整块体侧面顶面与底座'],
'C39':['后排侧面顶面可辨，下部遮挡','侧面底座可辨，右下遮挡','完整侧面顶面底座','侧壁可辨，杆挡左半','主体和附件可辨，杆遮右半','近景主体顶面三柱可辨，底缘截断','宽侧壁底座可辨，下角遮挡','圆柱大部曲面可辨，下部前景遮挡'],
'C40':['后方面板大部可辨，下部遮挡','完整侧面面板与底座','!右图缘仅窄侧壁底座，附件未见','完整柜体面板侧面及底座','完整圆柱与底座'],
}

def validate(e,ds):
    labels={l['event_id']:l for f in e['events'] for l in f['labels']}
    if len(ds)!=len(labels) or {d['event_id'] for d in ds}!=set(labels):raise ValueError('Missing or duplicate decision')
    for d in ds:
        l=labels[d['event_id']]
        if d['crop_sha256']!=l['crop_sha256'] or not d['reason']:raise ValueError('Stale or empty decision')
        if d['pixel_visibility_certified'] is not False or d['training_approved'] is not False:raise ValueError('Unsupported approval')

def main():
    e=read(OUT/'evidence.json');verify(e);ds=[];risk=[]
    if set(OBS)!={f['event_id'] for f in e['events']}:raise ValueError('Observation scope mismatch')
    for f in e['events']:
        notes=OBS[f['event_id']]
        if len(notes)!=len(f['labels']):raise ValueError('Label review count mismatch '+f['event_id'])
        for l,note in zip(f['labels'],notes):
            insufficient=note.startswith('!')
            ds.append(dict(event_id=l['event_id'],member_id=f['member']['member_id'],object_id=l['object_id'],truth=l['truth'],
                crop_sha256=l['crop_sha256'],image_sha256=f['member']['image_sha256'],label_sha256=f['member']['label_sha256'],
                status='insufficient_or_uncertain' if insufficient else 'identifiable_geometry_with_recorded_limits',reason=note.lstrip('!'),
                review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),pixel_visibility_certified=False,training_approved=False,
                scope='Visual content review with known source; not an independent class-recognition test or instance-mask certification.'))
            if insufficient:risk.append(f['member']['member_id'])
    validate(e,ds)
    frozen(OUT/'review.json',dict(status='full_label_review_complete_new_risk_caps_required',decisions=ds,new_risk_members=sorted(set(risk)),
        whole_image_screens_not_reused_as_box_approvals=True,training_started=False,
        inputs={str(x):file_sha256(x) for x in [OUT/'evidence.json',Path(__file__)]}))
    print('REVIEWED',len(ds),'LABELS; NEW_RISK_IMAGES',len(set(risk)))

if __name__=='__main__':main()
