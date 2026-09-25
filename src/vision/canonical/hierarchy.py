"""Separate renderer grouping without changing static equipment world geometry."""
import hashlib
import xml.etree.ElementTree as ET


def _pose(model):
    element=model.find('pose')
    if element is not None and element.get('relative_to'):
        raise ValueError('Explicit frame references require a frame resolver')
    values=[float(v) for v in model.findtext('pose','0 0 0 0 0 0').split()]
    if len(values)!=6:
        raise ValueError('Invalid model pose')
    return values


def flatten_equipment(world, object_names):
    """Support the canonical translation-only static parent containers, fail closed otherwise."""
    requested=set(object_names)
    found={}
    def visit(parent, ancestors):
        for model in parent.findall('model'):
            if model.get('name') in requested:
                name=model.get('name')
                if name in found:raise ValueError('Ambiguous equipment name')
                found[name]=(parent,model,ancestors)
            visit(model,ancestors+[model])
    visit(world,[])
    if set(found)!=requested:raise ValueError('Equipment missing from world')
    changes=[]
    # Validate and calculate all moves before changing the tree.
    for name,(parent,model,ancestors) in sorted(found.items()):
        if parent is world:continue
        if model.find('model') is not None:raise ValueError('Nested equipment hierarchy unsupported')
        pose=_pose(model)
        for ancestor in ancestors:
            values=_pose(ancestor)
            if any(abs(v)>1e-12 for v in values[3:]):raise ValueError('Rotated container requires frame composition')
            if ancestor.findtext('static','false')!='true':raise ValueError('Only static containers may be flattened')
            pose[:3]=[a+b for a,b in zip(pose[:3],values[:3])]
        if model.findtext('static','false')!='true':raise ValueError('Only static equipment may be flattened')
        payload=b''.join(ET.tostring(child) for child in model if child.tag!='pose')
        changes.append((name,parent,model,pose,hashlib.sha256(payload).hexdigest()))
    result=[]
    for name,parent,model,pose,payload_hash in changes:
        old=_pose(model)
        element=model.find('pose')
        if element is None:element=ET.SubElement(model,'pose')
        element.text=' '.join(format(v,'.17g') for v in pose)
        parent.remove(model);world.append(model)
        result.append({'object_id':name,'old_local_pose':old,'world_pose':pose,
                       'new_local_pose':pose,'unchanged_non_pose_payload_sha256':payload_hash})
    return result
