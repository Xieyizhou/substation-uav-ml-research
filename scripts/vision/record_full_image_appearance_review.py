"""Explicit observations of sixteen independently viewed variant overlays."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.capture_full_image_appearance import OUT,read,save,file_sha256,verify_tree
from scripts.vision.record_full_image_pilot_review import validate_observations

OBS={
 'F01-steel_original':{'capacitor_east':'钢灰宽主体及顶面、基座可辨，右侧杆遮挡仍存在。','transformer_se':'灰色主体上半部、顶部三附件及右侧可辨，下部被蓝色普通柜挡住。'},
 'F02-steel_original':{'capacitor_east':'灰色宽主体连续、基座完整；斜阴影没有遮没轮廓。'},
 'F03-steel_original':{'entry_switchgear':'灰色背侧箱体、顶面和基座清晰；没有前面板证据。'},
 'F04-steel_original':{'reactor_north':'灰色圆柱侧面有明暗渐变，顶面、完整基座可辨。'},
 'F05-steel_original':{'reactor_north':'较低角度的灰色圆柱轮廓、连续主体和基座可辨。'},
 'F06-steel_original':{'transformer_se':'灰色箱体、顶面三附件和基座完整可见。','capacitor_east':'左侧灰色主体和顶面可见；蓝色前景柜遮住左下角。'},
 'F07-steel_original':{'transformer_se':'灰色连续主体、三个顶部附件与基座清楚。','capacitor_east':'后方灰色柜右下被挡，左侧宽主体和顶面仍可辨。'},
 'F08-steel_original':{'entry_switchgear':'灰色背侧主体完整、顶面与基座可辨，无前面板；与 F03 同源。'},
 'F01-steel_warm_dim':{'capacitor_east':'暖暗条件主体较暗，但宽表面及顶面边界可辨；杆挡住右缘。','transformer_se':'暗色主体上半部、顶部附件和右边缘可辨，蓝柜遮挡下部。'},
 'F02-steel_warm_dim':{'capacitor_east':'暗色宽主体连续，斜阴影对比变弱，基座及外轮廓仍可辨。'},
 'F03-steel_warm_dim':{'entry_switchgear':'主体明显变暗，但背侧两面、较亮顶面和基座可分；面板不可见。'},
 'F04-steel_warm_dim':{'reactor_north':'暗色圆柱侧面渐变、顶面边界及基座可辨；不是仅黑色无内容框。'},
 'F05-steel_warm_dim':{'reactor_north':'暗色圆柱完整轮廓及右侧渐变可辨，主体与基座仍可分。'},
 'F06-steel_warm_dim':{'transformer_se':'暗色主体和较亮顶面分界清楚，三个附件可见，基座可辨。','capacitor_east':'暗色宽主体与较亮顶面可辨，左下前景柜遮挡仍在。'},
 'F07-steel_warm_dim':{'transformer_se':'暗主体、顶面、顶部三个附件及基座可辨，无严重截断。','capacitor_east':'后方暗色宽主体左段及顶面可辨，右下被变压器遮挡。'},
 'F08-steel_warm_dim':{'entry_switchgear':'暗色背侧主体、较亮顶面和基座轮廓可辨；无面板证据，与 F03 近邻同源。'},
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
    save(dest,dict(status='16_frames_visually_reviewed_pending_variant_dedup',frames=frames,
      frame_count=16,box_count=sum(len(f['objects']) for f in frames),
      next_required=['Variant-level file/pixel/protected-fingerprint exclusion; register all designed siblings.',
       'Complete remaining ochre/sage matrix only after intake checks; never use these 24 frames as 24 independent scenes.',
       'Freeze development dataset and exposure protocol before training.'],
      inputs={str(x):file_sha256(x) for x in (p,Path(__file__))}))
    print('REVIEW_RECORDED',16,sum(len(f['objects']) for f in frames),flush=True)

if __name__=='__main__':main()
