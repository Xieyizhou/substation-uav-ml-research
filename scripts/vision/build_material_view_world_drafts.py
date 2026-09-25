"""Verified original/warm/cool world drafts, not captured/admitted images."""
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.establish_material_view_candidates import OUT as SOURCE, prior
from scripts.vision.prepare_visual_augmentation_batch import SOURCE as ORIGINAL_PLAN

OUT=SOURCE/'world-drafts-v1'
PALETTES={'warm':'0.55 0.48 0.36 1','cool':'0.25 0.28 0.32 1'}


def valid_clock(message):
    """Versioned repair; accepts protobuf integer strings, not missing or fractional seconds."""
    try:
        stamp=message['header']['stamp']
        sec=float(stamp.get('sec',stamp.get('seconds')))
        nano=float(stamp.get('nsec',stamp.get('nanoseconds',0)))
        return math.isfinite(sec) and sec>=1 and sec.is_integer() and math.isfinite(nano) and 0<=nano<1e9 and nano.is_integer()
    except (KeyError,TypeError,ValueError):return False


def signature(node):
    return (node.tag,tuple(sorted(node.attrib.items())),(node.text or '').strip(),tuple(signature(c) for c in node))


def material_nodes(tree, names, visuals=('body','front_panel','reactor')):
    result={}
    for model in tree.findall('./world/model'):
        if model.get('name') not in names:continue
        for link in model.findall('link'):
            for visual in link.findall('visual'):
                if visual.get('name') not in visuals:continue
                for tag in ('ambient','diffuse'):
                    node=visual.find('material/'+tag)
                    if node is None:raise ValueError('Missing material field')
                    key=(model.get('name'),link.get('name'),visual.get('name'),tag)
                    if key in result:raise ValueError('Duplicate material identity')
                    result[key]=node
    return result


def check_only_materials(a,b,names,visuals=('body','front_panel','reactor')):
    a=copy.deepcopy(a);b=copy.deepcopy(b)
    an=material_nodes(a,names,visuals);bn=material_nodes(b,names,visuals)
    if not an or set(an)!=set(bn):raise ValueError('Material membership changed')
    changes=[]
    for key in an:
        if an[key].text!=bn[key].text:changes.append(list(key))
        an[key].text=bn[key].text='ALLOWED_RGB_FIELD'
    if signature(a)!=signature(b):raise ValueError('Non-allowed world change')
    return changes


def main():
    rp=SOURCE/'source-review.json';cp=SOURCE/'source-review-completion-v1.json'
    r=prior.read(rp);c=prior.read(cp)
    for x in (r,c,prior.read(SOURCE/'source-inventory.json')):prior.verify(x)
    original=prior.read(ORIGINAL_PLAN);wp=ORIGINAL_PLAN.parent/'world.sdf'
    if prior.file_sha256(wp)!=original['files']['world.sdf']:raise ValueError('Original world hash changed')
    tree=ET.parse(wp).getroot()
    boxes=tree.findall('.//sensor[@type="boundingbox_camera"]/camera/box_type')
    if len(boxes)!=1:raise ValueError('Missing/duplicate camera')
    boxes[0].text='full_2d'  # Retain current semantic mode, do not restore historical visible_2d.
    OUT.mkdir(exist_ok=True);rows=[];paths=[rp,cp,ORIGINAL_PLAN,wp,Path(__file__)]
    eligibility={f['source_review_id']:f['status'] for f in c['frames']}
    for s in r['sources']:
        if eligibility[s['source_review_id']]!='visual_content_reviewed_only':continue
        saved=ET.parse(s['source_world']).getroot()
        names={v['object_id'] for v in s['instance_mapping'].values()}
        restoration=check_only_materials(tree,saved,names)
        for variant in ('original',*PALETTES):
            world=copy.deepcopy(tree);changes=[]
            if variant!='original':
                nodes=material_nodes(world,{s['object_id']},('body','reactor'))
                if len(nodes)!=2:raise ValueError('Expected one target body with ambient/diffuse')
                for n in nodes.values():n.text=PALETTES[variant]
                changes=check_only_materials(tree,world,{s['object_id']},('body','reactor'))
                if len(changes)!=2:raise ValueError('Target material not actually changed')
            path=OUT/(s['source_review_id']+'-'+variant+'.sdf')
            if path.exists():
                if signature(ET.parse(path).getroot())!=signature(world):raise ValueError('Existing draft differs')
            else:ET.ElementTree(world).write(path,encoding='utf-8',xml_declaration=True)
            paths.append(path)
            rows.append(dict(source_review_id=s['source_review_id'],variant=variant,world_path=str(path),
                target_object_id=s['object_id'],actual_carrier_pose=s['actual_pose'],source_lineage=s['lineage_id'],
                restoration_fields=restoration,variant_fields=changes,annotation_mode='full_2d',
                image_exists=False,full_label_review_pending=True,training_ready=False))
    prior.frozen(OUT/'manifest.json',dict(status='six_world_drafts_only_capture_blocked',rows=rows,
        source_control_warning='Original appearance restored from hash-verified ancestor while retaining full_2d; no original RGB exists at these poses in this draft bundle.',
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('DRAFT_WORLDS',len(rows),'IMAGES 0; NO_TRAINING')


if __name__=='__main__':main()
