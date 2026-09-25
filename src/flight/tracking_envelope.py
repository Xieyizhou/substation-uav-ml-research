"""Conservative horizontal tracking-tube checks; no vehicle connection.

An L-infinity box encloses the permitted Euclidean cross-track displacement.
Closed-boundary intersection is a rejection, matching the existing map rule.
This does not certify map accuracy, hidden geometry, or aircraft safety.
"""
import math


def segment_intersects_box(a,b,box,padding):
    values=(*a,*b,padding,box['x0'],box['x1'],box['y0'],box['y1'])
    if not all(math.isfinite(v) for v in values) or padding<0:
        raise ValueError('Invalid segment/envelope input')
    if box['x0']>box['x1'] or box['y0']>box['y1']:
        raise ValueError('Inverted obstacle bounds')
    enter,leave=0.,1.
    for axis,lo,hi in ((0,box['x0']-padding,box['x1']+padding),(1,box['y0']-padding,box['y1']+padding)):
        delta=b[axis]-a[axis]
        if abs(delta)<1e-15:
            if a[axis]<lo or a[axis]>hi:return False
            continue
        t0,t1=(lo-a[axis])/delta,(hi-a[axis])/delta
        enter=max(enter,min(t0,t1));leave=min(leave,max(t0,t1))
        if enter>leave:return False
    return True


def check_route(points,boxes,*,clearance=1.8,tracking_error=.18,map_uncertainty=0.):
    if len(points)<2:raise ValueError('Route needs at least two points')
    if not all(math.isfinite(v) and v>=0 for v in (clearance,tracking_error,map_uncertainty)):
        raise ValueError('Invalid error budget')
    if any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('Invalid route coordinates')
    # Map boundary has no obstacle-clearance term; keep the entire tracking
    # tube inside the same approved [0.5,19.5] planning region.
    margin=tracking_error+map_uncertainty
    violations=[]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        if any(not .5+margin<v<19.5-margin for p in (a,b) for v in p):
            violations.append(dict(segment=i,obstacle='approved_map_boundary'))
        for box in boxes:
            if segment_intersects_box(a,b,box,clearance+margin):
                violations.append(dict(segment=i,obstacle=box.get('name','unnamed')))
    return dict(passed=not violations,clearance_m=clearance,tracking_error_m=tracking_error,map_uncertainty_m=map_uncertainty,required_centerline_padding_m=clearance+margin,violations=violations,interpretation='conditional geometric check only; map uncertainty is an assumption, not a certified bound')
