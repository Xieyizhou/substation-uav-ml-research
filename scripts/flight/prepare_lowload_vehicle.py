"""Independent flight-only RGB profile; never changes canonical sensors."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.flight.fly_px4_shadow_hover import ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'
OUT=BASE/'lowload-vehicle-v1'
MODEL='x500_research_lowload'

def verify_change(original, changed):
    restored=copy.deepcopy(changed)
    model=restored.getroot().find('model')
    if model.get('name')!=MODEL:raise ValueError('Unexpected model identity')
    model.set('name',original.getroot().find('model').get('name'))
    sensor=model.find("link[@name='research_camera_link']/sensor[@name='research_rgb']")
    old=original.getroot().find("model/link[@name='research_camera_link']/sensor[@name='research_rgb']")
    for field,expected in [('width','640'),('height','360')]:
        element=sensor.find('camera/image/'+field)
        if element.text!=expected:raise ValueError('Unexpected RGB resolution')
        element.text=old.find('camera/image/'+field).text
    if ET.tostring(restored.getroot())!=ET.tostring(original.getroot()):
        raise ValueError('Non-allowed vehicle/sensor change')

def prepare():
    source=ROOT/'simulation/models/x500_research/model.sdf'
    original=ET.parse(source);tree=copy.deepcopy(original)
    tree.getroot().find('model').set('name',MODEL)
    sensor=tree.getroot().find("model/link[@name='research_camera_link']/sensor[@name='research_rgb']")
    sensor.find('camera/image/width').text='640';sensor.find('camera/image/height').text='360'
    verify_change(original,tree)
    model=OUT/'models'/MODEL;model.mkdir(parents=True,exist_ok=False)
    tree.write(model/'model.sdf',encoding='utf-8',xml_declaration=True)
    verify_change(original,ET.parse(model/'model.sdf'))
    config=ET.Element('model');ET.SubElement(config,'name').text=MODEL;ET.SubElement(config,'version').text='1'
    ET.SubElement(config,'sdf',version='1.9').text='model.sdf';ET.ElementTree(config).write(model/'model.config')
    paths=[source,model/'model.sdf',model/'model.config',Path(__file__).resolve()]
    write_record(OUT/'protocol.json',dict(status='flight_profile_frozen_not_flight_verified',only_changes=['model resource name','RGB resolution 1920x1080 to 640x360'],unchanged=['geometry','inertia','lidar','depth','bounding-box sensor','camera pose/FOV/rate'],canonical_collection_allowed=False,vision_control_authority='none',limitation='RGB and box sensor resolutions differ; not a canonical collection profile or detector-quality validation',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))

if __name__=='__main__':prepare()
