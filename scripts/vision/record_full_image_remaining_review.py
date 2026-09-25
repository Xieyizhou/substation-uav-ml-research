"""Fixed observations after separately viewing all 32 full-image overlays."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.capture_full_image_remaining import OUT,read,save,file_sha256,verify_tree
from scripts.vision.record_full_image_pilot_review import validate_observations

OBS={
 'F01-ochre_original':{'capacitor_east':'赭色宽主体及顶面可见，右缘被杆挡住；基座连续。','transformer_se':'赭色上部箱体、三个顶部附件及右侧可见，下部被蓝色普通柜遮挡。'},
 'F02-ochre_original':{'capacitor_east':'赭色宽主体完整，斜阴影和基座可辨。'},
 'F03-ochre_original':{'entry_switchgear':'赭色背侧两面、明亮顶面及基座清楚，无前面板。'},
 'F04-ochre_original':{'reactor_north':'赭色圆柱顶面、渐变侧面、基座完整可见。'},
 'F05-ochre_original':{'reactor_north':'低视角赭色圆柱连续主体、右侧明暗渐变与基座清楚。'},
 'F06-ochre_original':{'transformer_se':'赭色宽主体、顶面三附件与基座完整。','capacitor_east':'后左赭色宽主体与顶面清楚，仅左下角被蓝柜遮挡。'},
 'F07-ochre_original':{'transformer_se':'赭色正侧面、三附件及基座可辨。','capacitor_east':'后方赭色左侧宽主体和顶面可见，右下被变压器挡住。'},
 'F08-ochre_original':{'entry_switchgear':'赭色背侧完整主体、顶面和基座可见，无面板；与 F03 同源。'},
 'F01-ochre_warm_dim':{'capacitor_east':'暗赭色主体连续，杆挡右缘，顶面及基座仍可辨。','transformer_se':'暗赭色上半主体、顶面附件和右侧可见，下部蓝柜遮挡。'},
 'F02-ochre_warm_dim':{'capacitor_east':'暗赭色宽主体和斜阴影可辨，完整基座与轮廓清楚。'},
 'F03-ochre_warm_dim':{'entry_switchgear':'暗赭色背侧与较亮顶面可分，基座完整；面板不在可见侧。'},
 'F04-ochre_warm_dim':{'reactor_north':'暖暗条件圆柱侧面渐变和顶面边界可辨，主体完整。'},
 'F05-ochre_warm_dim':{'reactor_north':'暗赭色圆柱侧面及右侧较亮渐变可见，基座清楚。'},
 'F06-ochre_warm_dim':{'transformer_se':'暗赭色箱体与亮顶面分界清楚，三个附件可辨。','capacitor_east':'暗赭色宽主体和顶面可见，左下仍被蓝柜遮挡。'},
 'F07-ochre_warm_dim':{'transformer_se':'暗赭色主体、亮顶面与三附件清楚，基座可辨。','capacitor_east':'后方暗赭色左侧主体与顶面清楚，右下遮挡。'},
 'F08-ochre_warm_dim':{'entry_switchgear':'暗赭色背侧连续、顶面边界及基座可见，无面板，近邻同源。'},
 'F01-sage_original':{'capacitor_east':'灰绿色宽主体与顶面可辨，杆遮挡右缘，基座可见。','transformer_se':'灰绿上半主体、三个顶部附件和右側可见，前景蓝柜挡住下部。'},
 'F02-sage_original':{'capacitor_east':'灰绿色连续宽主体、斜阴影和完整基座清楚。'},
 'F03-sage_original':{'entry_switchgear':'灰绿背侧两面和浅色顶面清楚，基座完整；无前面板。'},
 'F04-sage_original':{'reactor_north':'灰绿圆柱侧面渐变、浅色顶面及基座完整。'},
 'F05-sage_original':{'reactor_north':'低视角灰绿圆柱全高可见，右侧渐变与基座清楚。'},
 'F06-sage_original':{'transformer_se':'灰绿箱体主体、浅色顶面三附件和基座完整。','capacitor_east':'灰绿宽主体与顶面清楚，仅左下角被蓝柜遮挡。'},
 'F07-sage_original':{'transformer_se':'灰绿主体与顶面分界清楚，三附件及基座可见。','capacitor_east':'灰绿后方左侧主体和顶面可辨，右下遮挡未覆盖全部主体。'},
 'F08-sage_original':{'entry_switchgear':'灰绿背侧主体、浅顶面与基座清楚，没有前面板，F03 同源。'},
 'F01-sage_warm_dim':{'capacitor_east':'暗绿连续主体、较亮顶面与基座可辨，右缘被杆遮挡。','transformer_se':'暗绿上半主体、顶面附件及右侧可辨，下部蓝柜遮挡。'},
 'F02-sage_warm_dim':{'capacitor_east':'暗绿宽主体和斜阴影仍可辨，完整外轮廓与基座清楚。'},
 'F03-sage_warm_dim':{'entry_switchgear':'暗绿背侧主体与较亮顶面可分，基座可见；无面板。'},
 'F04-sage_warm_dim':{'reactor_north':'暗绿圆柱渐变侧面、顶面边界及完整基座可辨。'},
 'F05-sage_warm_dim':{'reactor_north':'暗绿圆柱连续全高、右侧亮度渐变和基座可辨。'},
 'F06-sage_warm_dim':{'transformer_se':'暗绿主体、亮顶面、三个顶部附件和基座清楚。','capacitor_east':'暗绿宽主体和顶面可辨，前景蓝柜挡左下角。'},
 'F07-sage_warm_dim':{'transformer_se':'暗绿连续箱体、较亮顶面三附件及基座可辨。','capacitor_east':'后方暗绿左侧宽主体和顶面可见，右下被挡。'},
 'F08-sage_warm_dim':{'entry_switchgear':'暗绿背侧完整、较亮顶面及基座可辨，无面板，保留近邻同源关系。'},
}

def main():
    dest=OUT/'review/decisions.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    p=OUT/'review/manifest.json';verify_tree(p);m=read(p);validate_observations(m['frames'],OBS)
    if any(f['pair_status']!='aligned' for f in m['frames']):raise ValueError('Pair held')
    frames=[]
    for f in m['frames']:
        frames.append({**f,'review_status':'visual_review_only_pending_dedup','review_nature':'AI辅助审核',
          'reviewed_at':datetime.now(timezone.utc).isoformat(),
          'objects':[{**o,'review_status':'visible_content_observed','reason':OBS[f['frame_id']][o['object_id']],
            'pixel_instance_certified':False,'component_mask_status':'unknown'} for o in f['objects']]})
    save(dest,dict(status='32_frames_visually_reviewed_pending_variant_dedup',frames=frames,
      frame_count=32,box_count=sum(len(f['objects']) for f in frames),
      limitations=['Not pixel-mask certification or automatic training admission.','Switchgear front panels remain out of view; same map/assets.'],
      inputs={str(x):file_sha256(x) for x in (p,Path(__file__))}))
    print('REVIEW_RECORDED',32,sum(len(f['objects']) for f in frames),flush=True)

if __name__=='__main__':main()
