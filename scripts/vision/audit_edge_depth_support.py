"""Non-admitting depth/geometry corroboration; never edits source labels."""
import math
import numpy as np
from scripts.vision.audit_edge_label_sources import (
    OUT as SOURCE, SOURCES, BASE, ROOT, read, save, file_sha256,
    verify_tree, ET, pose, rotate, Path,
)

OUT = SOURCE / 'depth-support-v1'


def ray_box(origin, directions, low, high):
    """Positive ray entry distance in the supplied (not normalized) ray units."""
    parallel = np.abs(directions) < 1e-12
    safe = np.where(parallel, 1., directions)
    a = (low - origin) / safe
    b = (high - origin) / safe
    entry = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1)
    leave = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1)
    outside = (parallel & ((origin < low) | (origin > high))).any(axis=-1)
    return np.where((leave >= np.maximum(entry, 0)) & ~outside,
                    np.maximum(entry, 0), np.inf)


def main():
    verify_tree(SOURCE / 'completion.json')
    prior = read(SOURCE / 'audit.json')
    inputs = {str(p): file_sha256(p) for p in
              (SOURCE / 'completion.json', SOURCE / 'audit.json', Path(__file__))}
    results = []
    for (rid, folder, view), audited in zip(SOURCES, prior['results']):
        assert rid == audited['review_id']
        receipt = read(BASE / folder / 'capture' / f'{view}.json')
        dp = Path(receipt['depth_path'])
        if file_sha256(dp) != receipt['depth_sha256']:
            raise ValueError('Stale depth')
        inputs[str(dp)] = file_sha256(dp)
        world = ET.parse(audited['source_world'])
        camera = next(m for m in world.iter('model') if m.get('name') == 'canonical_camera')
        link = camera.find("link[@name='research_camera_link']")
        rgb, depth = [link.find(f"sensor[@name='research_{name}']") for name in ('rgb', 'depth')]
        if pose(rgb) != [0., 0., 0.] or pose(depth) != [0., 0., 0.]:
            raise ValueError('Nonzero sensor offset')
        if rgb.findtext('camera/horizontal_fov') != depth.findtext('camera/horizontal_fov'):
            raise ValueError('Unaligned FOV')
        w, h = [int(depth.findtext('camera/image/' + k)) for k in ('width', 'height')]
        if (w, h) != (640, 360):
            raise ValueError('Unexpected depth dimensions')
        observed = np.fromfile(dp, dtype='<f4').reshape(h, w)
        f = w / (2 * math.tan(float(depth.findtext('camera/horizontal_fov')) / 2))
        yy, xx = np.mgrid[:h, :w]
        rays = np.stack((np.ones_like(xx), -(xx + .5 - w/2)/f,
                         -(yy + .5 - h/2)/f), axis=-1)
        q = receipt['actual_pose']['orientation']
        rotation = np.array([rotate(q, v) for v in np.eye(3)]).T
        directions = rays @ rotation.T
        center = np.array(audited['optical_center'])
        model = next(m for m in world.iter('model') if m.get('name') == audited['scene_object_id'])
        components = []
        for l in model.findall('link'):
            for v in l.findall('visual'):
                dims = np.array(list(map(float, v.findtext('geometry/box/size').split())))
                vc = np.array(pose(model)) + pose(l) + np.array(pose(v))
                expected = ray_box(center, directions, vc-dims/2, vc+dims/2)
                hit = np.isfinite(expected) & (expected > .1) & (expected < 120)
                finite = hit & np.isfinite(observed)
                metrics = {}
                for convention, values in [('axial', expected), ('range', expected*np.linalg.norm(rays, axis=-1))]:
                    residual = np.abs(observed[finite] - values[finite])
                    metrics[convention] = {
                        'median_absolute_residual_m': float(np.median(residual)) if residual.size else None,
                        'within_002m': int((residual <= .02).sum()),
                        'within_005m': int((residual <= .05).sum()),
                    }
                components.append(dict(name=v.get('name'),potential_depth_pixel_centers=int(hit.sum()),
                                       finite_observed=int(finite.sum()), residuals=metrics))
        results.append(dict(review_id=rid,components=components,
                            rgb_depth_skew_ms=abs(receipt['rgb_timestamp']-receipt['depth_timestamp'])*1000,
                            decision='hold_no_relabel',visibility='unknown'))
    save(OUT / 'audit.json', dict(status='corroboration_complete_not_visibility_certified',inputs=inputs,
        results=results, pixel_center_convention='.5',
        limitations=['Depth is not an instance mask; coincident surfaces may agree.',
                     'Both axial and range conventions reported; no convention selected for admission.',
                     'Low-resolution samples do not certify full-resolution visible area.',
                     'Saved static box geometry only; no renderer replay or occlusion segmentation.'],
        training_admitted=False,promotable=False))
    print(results)


if __name__ == '__main__':
    main()
