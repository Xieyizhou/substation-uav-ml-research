"""Import explicit observations from the 24 inspected, prediction-free pages.

This is a review import, never a generator of training approvals. Each line is a
human-readable observation entered after visual inspection in the active task.
OMBL status order is explicit; equal statuses are not inferred from geometry.
"""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.material_control_feasibility import OUT, prior
from src.ml.artifacts import object_sha256

# view | object | original/material/background/lighting decisions | observation
# | original appearance | material appearance | background appearance | lighting appearance
OBSERVATIONS = '''
01|capacitor_east|PPPP|主体较宽，下部被前景 cabinet_center 遮挡，不能把前景柜的面板算给目标|青灰顶面和侧面可分|灰主体与前景蓝柜可分|暗天空下顶边仍可分|深色主体仍有宽侧面
01|transformer_se|PPPP|右图缘截断，宽主体与顶部端子仍可辨|蓝黑主体与端子可辨|灰主体仍保留端子|天空变暗但端子轮廓仍在|主体更暗，端子仍分离
02|reactor_north|CCCC|圆柱主体及基座完整，未见明显前景遮挡|深灰圆柱边缘完整|灰圆柱明暗与底座可分|蓝灰天空下主体边缘仍清楚|暗圆柱与基座边界仍在
02|switchgear_west|PPPP|左下图缘截断，较宽顶面和侧面可辨，未朝向面板|青色顶侧面可辨|灰色顶侧面仍可分|顶侧面与褐地分离|蓝色顶侧面明暗可辨
03|reactor_north|PPPP|仅上段圆柱露在前景柜之后，宽度可辨但下部与基座被挡|灰上段与蓝柜分离|灰圆柱与灰柜靠明暗分离|上缘与天空较弱但仍在|暗圆柱上段轮廓可见
03|switchgear_west|PPPP|主体及面板大部可辨，左下被近柜遮挡|深面板有明显边界|面板与主体近同灰，仅细边界|深面板仍与蓝框分离|暗面板轮廓仍明显
03|transformer_mid|PPPP|右图缘截断，宽侧面及三端子可辨|蓝黑两侧面可辨|灰两侧面及端子可辨|端子与天空对比较弱|暗主体仍见转角及端子
03|west_switchgear_02|PPPP|近景柜下部出画，宽顶面、上部面板存在|青顶与深面板可分|面板大部融入灰主体，细上边可见|青顶与深面板仍分离|深色面板上边仍可见
03|west_switchgear_03|FFFF|后柜仅顶部和窄前面露出，前景顶面占裁剪大部|窄蓝条和面板边线|灰窄条，难辨面板内容|窄蓝条仍可见但内容不足|暗蓝窄条内容不足
03|west_switchgear_04|FFFF|最远柜只剩顶盖及窄带，被多层前柜遮挡|顶盖与窄蓝带|灰顶盖及窄暗带|顶盖与天空边界存在|暗顶盖与窄带，不能视作完整柜
04|capacitor_east|PPPP|后方块体顶面与上半侧面可见，底部被多层前景挡住|蓝灰后块可分|灰后块与蓝前柜可分|后块与天空边线仍在|后块暗面仍可分
04|transformer_mid|CCCC|近景完整宽主体、三端子和基座可辨，主体贴近下缘但未明显截断|蓝黑两面及浅端子|灰两面及浅端子|与地面和阴影分离|暗主体转角及端子仍清楚
04|transformer_se|PPPP|远处主体下部被近变压器遮住，三个端子及宽侧面在|蓝黑宽侧面|灰侧面与近景靠边界分离|浅端子在天空上较弱|暗宽侧面与端子可辨
04|transformer_sw|PPPP|右缘截断，宽侧面、顶面和端子仍可辨|蓝黑侧面|灰側面和端子|天空上端子较弱但可见|暗侧面轮廓保留
05|switchgear_west|PPPP|后柜下半被近变压器挡住，顶部与面板上部可见|深面板与蓝框明显|灰面板细边可见，对比显著弱|蓝框面板分离|暗面板仍有蓝框
05|transformer_sw|PPPP|前景主体同时右缘及下缘出画，宽顶面和端子可见|宽蓝灰顶面|宽灰顶面浅端子|顶面与天空地面可分|暗顶面和浅端子
05|west_switchgear_01|CCCC|独立完整柜体，斜侧正面、顶部与基座均可见|蓝侧面与深面板|灰侧面及细面板边线仍可见|柜体轮廓与褐地分离|深面板和蓝侧面分离
05|west_switchgear_02|CCCC|主体及面板几乎完整，前景仅邻近裁剪下沿|蓝框和面板明确|灰面板细边明确但同色|面板边界保留|暗面板仍可辨
05|west_switchgear_03|PPPP|左部被前柜挡住，侧面、顶面与部分面板仍宽|蓝侧面及残余面板|灰柜相接，靠细面板边线分离|后柜侧面轮廓保留|深面板残部与侧面可分
05|west_switchgear_04|PPPP|左半及下部被柜和端子遮挡，右侧面与面板竖条可见|蓝框和窄面板|灰侧面仍见，面板竖边很弱|侧面蓝条仍辨识|蓝侧面与深窄面板
06|entry_switchgear|UUUU|裁剪大部为前景变压器，仅其后极窄顶边；无掩码不能认证该片段的视觉归属|细青边不足以辨柜|灰边与前景相接，无法确认|细青边仍内容不足|暗细边不足以确认
06|reactor_north|PPPP|右缘切掉圆柱部分，仍有宽圆柱侧面、顶弧和底座|灰圆柱曲面可辨|灰圆柱渐变可辨|主体与地面边界保留|暗圆柱和底座仍可辨
06|switchgear_west|PPPP|下部被近变压器挡住，柜背宽面及顶面存在|蓝背与浅顶面|灰背与灰顶面明暗分离|背面与地面可分|蓝背与亮顶分离
06|transformer_mid|PPPP|近变压器下部出画，宽顶面与三端子可见|蓝顶面和浅端子|灰顶面和浅端子|端子仍独立可见|暗顶面与端子可分
06|transformer_sw|PPPP|左缘截断并有前景杆遮挡，但宽主体与端子存在|蓝黑主体被黑杆穿过|灰主体与黑杆分离|主体和杆轮廓仍可辨|暗主体对黑杆较弱仍可分
06|west_switchgear_01|PPPP|左下被近变压器遮挡，柜背与右侧主体仍较宽|蓝背右侧面|灰背右侧明暗|背面与天空可分|蓝背和暗侧面可分
06|west_switchgear_02|CCCC|孤立完整背面、侧面、顶面与基座，无正面面板证据|蓝背面完整|灰背面完整|与褐地边界明确|蓝背与暗侧面明确
06|west_switchgear_03|PPPP|右下被另柜遮挡，顶面和大部背面仍在|蓝背与前蓝柜相接|灰背与前灰柜靠边界分离|大部背面仍可辨|蓝背大部可辨
06|west_switchgear_04|PPPP|左下被前柜遮住，右侧宽竖面与顶面可见|蓝竖面可辨|灰竖面明暗可辨|顶边仍分离|蓝竖面可辨但无面板
07|capacitor_east|PPPP|后块下部被变压器挡，左侧有杆，宽上部及顶面可见|蓝灰上部与黑杆|灰上部与黑杆分离|天空对比弱些但边缘存在|暗上部仍与杆可分
07|entry_switchgear|UUUU|仅远变压器右侧窄条，主体被遮，视觉归属不能独立确认|窄青侧条|窄灰条与灰前景融接|窄青条内容不足|窄蓝条内容不足
07|reactor_north|CCCC|完整圆柱、顶椭圆和基座独立可见|灰圆柱完整|灰圆柱顶侧明暗完整|圆柱与地面可分|暗圆柱顶侧及基座可分
07|switchgear_west|CCCC|完整柜背、顶面及基座，无正面面板证据|宽蓝背可辨|宽灰背明暗可辨|柜背与地面分离|暗蓝背和亮顶可辨
07|transformer_mid|PPPP|下部被近圆柱挡住，顶部端子及上侧面可辨|蓝顶和端子在圆柱后|灰顶与灰圆柱有重叠但端子在|端子与天空分离较弱|暗顶仍见浅端子
07|transformer_se|PPPP|后变压器被近变压器遮住下部，端子及上侧面可辨|后蓝侧面与端子|后灰侧面与端子|浅端子天空对比弱|深后侧面和端子可辨
07|transformer_sw|PPPP|右下被近柜小部分遮挡，大部侧面及端子仍在|蓝黑侧面|灰侧面和端子|浅端子轮廓仍在|暗侧面及端子可辨
07|west_switchgear_01|PPPP|列队后柜被多柜遮挡，顶部与左侧宽面可辨|蓝侧条与顶面|灰条状侧面可分|蓝侧与天空分离|蓝侧面明暗可分
07|west_switchgear_02|PPPP|被前柜遮挡且靠右缘，仍见较宽斜侧面和顶面|蓝斜侧和顶面|灰斜侧和顶面|蓝斜侧仍存在|暗蓝斜面仍存在
07|west_switchgear_03|PPPP|右缘截断及相邻遮挡，剩较宽斜侧面和基座角|蓝侧面与基座|灰侧面与基座|斜侧面与地面分离|暗侧面及基座角
07|west_switchgear_04|FFFF|最右缘仅窄竖侧片及基座角，内容不足|窄蓝边|窄灰边|窄蓝边与地面|窄暗蓝边，不能认证完整外形
08|capacitor_east|CCCC|完整宽侧面与基座，有斜阴影但无明显前景遮挡|蓝灰面与斜阴影|灰面与斜阴影可分|轮廓在暗天空前仍明确|暗蓝面上阴影仍可见
08|transformer_mid|UUUU|左缘极窄后方片段且被前柜遮挡，裁剪主要不是该目标，不能确认可辨内容|窄灰蓝顶片|灰顶片，前蓝柜未变|窄片段内容不足|暗窄片段无法确认
09|capacitor_east|PPPP|后方块体下部被近变压器及柜挡住，顶面与宽上侧面在|蓝灰后块|灰后块与蓝前柜|后块顶边仍可见|暗后块上面可辨
09|entry_switchgear|PPPP|左部被前变压器挡，右侧面、顶面和底座仍可辨|蓝右半与基座|灰右半与灰前景靠转角分离|蓝右侧面可见|深蓝右侧和基座
09|switchgear_west|CCCC|完整孤立背侧块体与基座，无面板正面|蓝顶背侧|灰顶背侧|轮廓与地面分离|暗背与蓝顶分离
09|transformer_mid|PPPP|左图缘截断，宽主体及三端子仍可辨|蓝黑主体和端子|灰主体和端子|端子天空对比较弱|暗主体仍见浅端子
09|transformer_se|PPPP|前杆遮挡右部，宽主体及顶端子仍在|黑杆跨蓝主体|黑杆跨灰主体|主体和杆仍分开|暗主体与杆对比弱但边界在
09|transformer_sw|CCCC|完整宽主体、三端子和基座可见|蓝黑两面|灰两面和端子|地面与底座分离|暗两面端子保留
09|west_switchgear_01|FFFF|最远柜被前柜遮住大部，仅窄顶面和上侧条|窄蓝条|窄灰条|窄蓝条仍在|窄蓝暗条内容不足
09|west_switchgear_02|FFFF|后柜被近柜挡住，仅顶面和窄侧条，右边贴图缘|顶面与窄蓝侧|顶面与窄灰侧|窄蓝侧保留|窄暗侧内容不足
09|west_switchgear_03|PPPP|下部被前柜遮住且右缘截断，顶面和一段宽侧面可见|蓝顶与侧面|灰顶与侧面|蓝侧面仍可分|暗侧面与亮顶
09|west_switchgear_04|PPPP|近柜右缘和下缘出画，宽顶侧面可辨|蓝顶侧面|灰顶侧面|与地面及天空分离|暗侧面和蓝顶面
10|entry_switchgear|CCCC|完整孤立柜体背侧、顶面、基座，无正面面板证据|蓝顶侧面|灰顶侧面明暗|与褐地和墙分离|暗蓝侧面及亮顶
11|capacitor_east|PPPP|左下被前景 cabinet_center 遮住一角，大部块体清楚|蓝灰主体可辨|灰主体与前蓝柜分离|顶边和主体保留|暗主体和前蓝柜分离
11|transformer_mid|PPPP|左缘及下缘截断，宽顶面与端子可辨|蓝顶面与端子|灰顶面与端子|端子仍可见|暗顶面与端子
11|transformer_se|PPPP|前杆横穿主体中右部，仍见大部侧面、顶部与端子|蓝黑主体与杆|灰主体与黑杆|主体轮廓保留|暗主体与杆边界较弱仍在
12|reactor_north|PPPP|圆柱下部被前变压器遮住，宽上侧面与顶椭圆可辨|灰圆柱上段|灰圆柱与灰前景靠曲边分离|圆柱顶边与天空较弱|暗圆柱顶侧明暗可辨
12|switchgear_west|PPPP|左缘截断，宽侧面和顶部、基座可见|蓝侧面与顶面|灰侧面明暗|与地面分离|蓝侧面与基座
12|transformer_mid|CCCC|完整主体、三端子与基座，前景不遮主体|蓝黑主体及端子|灰主体及端子|主体与地面分离|暗主体和端子仍明确
12|west_switchgear_04|FFFF|左缘后柜只露顶部和窄侧片，其余被前柜遮住|窄蓝片与顶盖|窄灰片与顶盖|窄蓝片保留|窄蓝暗片内容不足
'''

def validate_decisions(evidence, decisions):
    expected={r['evidence_id']:r for r in evidence}
    actual={r['evidence_id']:r for r in decisions}
    if len(actual)!=len(decisions) or actual.keys()!=expected.keys():
        raise ValueError('Missing/duplicate/unexpected review')
    for key,d in actual.items():
        r=expected[key]
        if d['evidence_sha256']!=object_sha256(r) or not d['reason']:
            raise ValueError('Stale evidence or empty decision')
        for field in ('image','crop'):
            if prior.file_sha256(r[field+'_path'])!=r[field+'_sha256']:
                raise ValueError('Stale visual evidence')
        if prior.file_sha256(d['page'])!=d['page_sha256']:
            raise ValueError('Stale review page')

def main():
    ep=OUT/'initial-gate.json';pp=OUT/'review-pages.json'
    e,p=prior.read(ep),prior.read(pp)
    for x in (e,p):prior.verify(x)
    views=sorted({r['view_id'] for r in e['corrected_target_records']})
    index={(r['view_id'],r['variant'],r['object_id']):r for r in e['corrected_target_records']}
    decisions=[];now=datetime.now(timezone.utc).isoformat()
    for line in OBSERVATIONS.strip().splitlines():
        number,oid,status,geometry,*notes=line.split('|')
        if len(status)!=4 or len(notes)!=4:raise ValueError('Incomplete explicit observation')
        for v,s,note in zip(('original','material','background','lighting'),status,notes):
            view=views[int(number)-1];r=index[view,v,oid]
            page=next(x for x in p['pages'] if x['view_id']==view and v in x['variants'])
            decisions.append(dict(evidence_id=r['evidence_id'],view_id=view,variant=v,object_id=oid,
                category=r['category'],pair_id=r['pair_id'],truth=r['truth'],
                stratum={'C':'clear','P':'partial','F':'fragment','U':'unknown'}[s],
                reason=geometry+'；本条件观察：'+note+'。',review_nature='AI辅助审核',reviewed_at=now,
                evidence_sha256=object_sha256(r),page=page['page'],page_sha256=page['sha256'],
                pixel_visibility_certified=False,training_approved=False))
    validate_decisions(e['corrected_target_records'],decisions)
    prior.frozen(OUT/'independent-review.json',dict(status='review_complete_with_named_unknowns',
        decisions=decisions,unknowns=[d['evidence_id'] for d in decisions if d['stratum']=='unknown'],
        definition='clear = broad discernible body without evident clipping/occlusion; not class certainty or training admission. partial/fragment are descriptive, no area threshold.',
        old_review_replaced=False,original_stratum_is_grouping_only=True,
        inputs={str(x):prior.file_sha256(x) for x in (ep,pp,Path(__file__).resolve())}))
    print('EXPLICIT_REVIEW',len(decisions),'unknown',sum(d['stratum']=='unknown' for d in decisions))

if __name__=='__main__':main()
