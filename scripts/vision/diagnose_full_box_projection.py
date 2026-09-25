"""Read-only analytical reproduction; never a renderer fix or label generator."""
import itertools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.vision.run_dual_box_diagnosis import OUT as PRIOR, SOURCE, ROOT, read, verify, frozen, file_sha256, ET
from scripts.vision.test_body_material_applicability import baseline_verify
from src.vision.canonical.plan import rotate

OUT = PRIOR / 'projection-probe-v1'
SOURCE_URL = 'https://raw.githubusercontent.com/gazebosim/gz-rendering/gz-rendering8_8.2.3/ogre2/src/Ogre2BoundingBoxCamera.cc'
SIGNS = list(itertools.product((-1, 1), repeat=3))


def legacy_reject(low, high):
    """Version 8.2.3 full-box endpoint-magnitude predicate, not clipping."""
    return bool(np.any((np.abs(low) > 1) & (np.abs(high) > 1)))


def outside_same_side(low, high):
    return bool(np.any((np.asarray(high) < -1) | (np.asarray(low) > 1)))


def near_clip_box(vertices, near):
    """Vertices of a convex box intersected with the forward half-space."""
    if near <= 0 or np.shape(vertices) != (8, 3) or not np.isfinite(vertices).all():
        raise ValueError('Invalid box or near plane')
    points = [p for p in vertices if p[0] >= near]
    for i, a in enumerate(SIGNS):
        for j in range(i + 1, 8):
            if sum(x != y for x, y in zip(a, SIGNS[j])) != 1:
                continue
            p, q = vertices[i], vertices[j]
            if (p[0] < near < q[0]) or (q[0] < near < p[0]):
                points.append(p + (q-p) * ((near-p[0])/(q[0]-p[0])))
    return np.asarray(points).reshape(-1, 3)


def project(vertices, hfov, aspect):
    if len(vertices) == 0 or np.any(np.abs(vertices[:, 0]) < 1e-12):
        raise ValueError('Empty projection or zero perspective denominator')
    scale = math.tan(hfov / 2)
    return np.column_stack((-vertices[:, 1]/vertices[:, 0]/scale,
                            vertices[:, 2]/vertices[:, 0]/scale/aspect))


def translation(element):
    pose = element.find('pose')
    if pose is not None and pose.attrib:
        raise ValueError('Unsupported relative pose')
    values = np.array([float(x) for x in element.findtext('pose', '0 0 0 0 0 0').split()])
    if len(values) != 6 or not np.isfinite(values).all() or np.any(values[3:] != 0):
        raise ValueError('Probe supports only zero-RPY object/link poses')
    return values[:3]


def reconstruct(frame):
    tree = ET.parse(frame['source_world'])
    models = tree.findall("./world/model[@name='west_switchgear_02']")
    if len(models) != 1:
        raise ValueError('Ambiguous instance model')
    model = models[0]
    if model.findall('model') or len(model.findall('link')) != 1:
        raise ValueError('Unsupported model hierarchy')
    camera = tree.find("./world/model[@name='canonical_camera']/link[@name='research_camera_link']")
    sensor = camera.find("sensor[@name='research_boxes']")
    if sensor.findtext('camera/box_type') != 'full_2d':
        raise ValueError('Unexpected box mode')
    q = frame['actual_pose']['orientation']
    rotation = np.column_stack([rotate(q, np.eye(3)[i]) for i in range(3)])
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError('Non-orthonormal camera rotation')
    optical_center = np.asarray(frame['actual_pose']['position']) + rotation @ (translation(camera)+translation(sensor))
    hfov = float(sensor.findtext('camera/horizontal_fov'))
    near = float(sensor.findtext('camera/clip/near'))
    width = int(sensor.findtext('camera/image/width'))
    height = int(sensor.findtext('camera/image/height'))
    rows = []
    for visual in model.findall('link/visual'):
        labels = visual.findall("plugin/label")
        if len(labels) != 1 or labels[0].text.strip() != '128':
            raise ValueError('Unexpected visual instance mapping')
        size = np.asarray([float(x) for x in visual.findtext('geometry/box/size', '').split()])
        if size.shape != (3,) or np.any(size <= 0):
            raise ValueError('Unsupported visual geometry')
        center = translation(model)+translation(model.find('link'))+translation(visual)
        world = np.array([center+size*np.array(s)/2 for s in SIGNS])
        camera_vertices = (world-optical_center) @ rotation
        ndc = project(camera_vertices, hfov, height/width)
        low, high = ndc.min(0), ndc.max(0)
        clipped = near_clip_box(camera_vertices, near)
        clipped_ndc = project(clipped, hfov, height/width) if len(clipped) else None
        rows.append(dict(visual=visual.get('name'), camera_vertices=camera_vertices.tolist(),
            depth_range=[float(camera_vertices[:, 0].min()), float(camera_vertices[:, 0].max())],
            raw_ndc_bounds=[low.tolist(), high.tolist()], legacy_rejected=legacy_reject(low, high),
            raw_same_side_rejected=outside_same_side(low, high),
            near_clipped_ndc_bounds=[clipped_ndc.min(0).tolist(), clipped_ndc.max(0).tolist()] if clipped_ndc is not None else None,
            near_clipped_extent_intersects_viewport=bool(clipped_ndc is not None and not outside_same_side(clipped_ndc.min(0), clipped_ndc.max(0)))))
    return dict(optical_center=optical_center.tolist(), hfov=hfov, near=near, resolution=[width,height], visuals=rows)


def main():
    paths = [PRIOR/'completion.json', PRIOR/'protocol.json', SOURCE/'protocol.json',
             PRIOR/'replay/T027/attempt-01/receipt.json']
    for path in paths:
        verify(read(path))
    frame = read(PRIOR/'protocol.json')['frames'][0]
    if frame['review_ids'] != ['T027']:
        raise ValueError('Unexpected frozen frame')
    verify(read(frame['source_receipt']))
    result = reconstruct(frame)
    if len(result['visuals']) != 3 or not all(r['legacy_rejected'] for r in result['visuals']):
        raise ValueError('Observed reconstruction differs; investigate instead of reusing report')
    tests = ['tests.test_full_box_projection', 'tests.test_dual_box_diagnosis', 'tests.test_edge_box_trace', 'tests.test_instance_visibility_diagnosis']
    run = subprocess.run([sys.executable, '-m', 'unittest', *tests], cwd=ROOT, capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise ValueError(run.stderr)
    baseline = baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40:
        raise ValueError('Baseline integrity failed')
    paths += [Path(frame['source_world']), Path(frame['source_receipt']), Path(__file__),
              ROOT/'src/vision/canonical/plan.py', ROOT/'docs/results/ml_full_box_projection_20260910.md',
              Path('/opt/homebrew/opt/gz-rendering8/.brew/gz-rendering8.rb'),
              Path('/opt/homebrew/opt/gz-rendering8/INSTALL_RECEIPT.json')]
    paths += [ROOT/(t.replace('.', '/')+'.py') for t in tests]
    OUT.mkdir(exist_ok=True)
    dest = OUT/'diagnosis.json'
    if dest.exists():
        verify(read(dest)); print('VALID_PROJECTION_DIAGNOSIS_REUSED'); return
    frozen(dest, dict(status='analytical_projection_failure_reproduced_runtime_branch_unconfirmed',
        source_url=SOURCE_URL, source_version='gz-rendering8_8.2.3',
        source_review='FullBoundingBoxes endpoint magnitude predicate; MeshMinimalBox divides projected vertices without near-plane clipping. Source review is not installed-binary instrumentation.',
        reconstruction=result, baseline=baseline,
        regression=dict(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr, whole_repository_tested=False),
        limits=['Pinhole reconstruction of SDF box corners, not captured Ogre VAO vertices or runtime projection matrices.',
                'Near-clipped extent is geometric, not visibility/occlusion evidence or replacement annotation.',
                'Runtime item visibility filter and exact branch not instrumented; no universal renderer-fix claim.'],
        training_ready=False, training_started=False, renderer_modified=False, historical_labels_changed=False,
        inputs={str(p):file_sha256(p) for p in paths}))
    print(run.stderr); print(result); print('PROJECTION_DIAGNOSIS_COMPLETE_TRAINING_HELD')


if __name__ == '__main__':
    main()
