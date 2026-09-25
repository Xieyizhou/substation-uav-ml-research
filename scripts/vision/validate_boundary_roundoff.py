"""Diagnostic-only float32 boundary normalization, never edits source labels."""
import math,struct
from pathlib import Path
from scripts.vision.audit_candidate29_completeness import OUT as AUDIT,prior,capture
from scripts.vision.candidate29_review_pages import raw_boxes

OUT=AUDIT.parent/'candidate29-unified-risk-policy-v1'

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]

def ulp32(x):
    if not math.isfinite(x) or x<=0:raise ValueError('Invalid image dimension')
    bits=struct.unpack('I',struct.pack('f',x))[0]
    return struct.unpack('f',struct.pack('I',bits+1))[0]-f32(x)

def normalize_box(box,width=1920,height=1080):
    if (width,height)!=(1920,1080):raise ValueError('Unvalidated sensor resolution')
    if len(box)!=4 or not all(math.isfinite(x) for x in box):raise ValueError('Invalid coordinates')
    if not(box[0]<box[2] and box[1]<box[3]):raise ValueError('Nonpositive raw area')
    tolerances=[ulp32(width),ulp32(height)]*2;limits=[width,height]*2;normalized=[];changes=[]
    for i,(x,bound,tol) in enumerate(zip(box,limits,tolerances)):
        y=min(max(x,0),bound)
        if abs(y-x)>tol:raise ValueError('Exceeds validated arithmetic boundary budget')
        normalized.append(y)
        if x!=y:changes.append(dict(coordinate=i,raw=x,normalized=y,delta=y-x,budget=tol))
    if not(normalized[0]<normalized[2] and normalized[1]<normalized[3]):raise ValueError('Degenerate normalized area')
    return dict(raw=list(box),normalized=normalized,changes=changes)

def main():
    OUT.mkdir(exist_ok=True);pp=AUDIT/'protocol.json';cp=AUDIT/'completion.json';bp=AUDIT/'review-pages/A16-boundary.json'
    for path in (pp,cp,bp):prior.verify(prior.read(path))
    f=next(f for f in prior.read(pp)['frames'] if f['pair_id']=='A16');b=prior.read(bp)
    source=capture.DEPTH/'source/ogre2/src/Ogre2BoundingBoxCamera.cc';dp=capture.DEPTH/'protocol.json';d=prior.read(dp);prior.verify(d)
    if prior.file_sha256(source)!=d['inputs'][str(source)]:raise ValueError('Renderer source changed')
    records=[]
    for r in b['records']:
        records.append(dict(index=r['index'],boxes={k:normalize_box(v) for k,v in r['raw_boxes'].items()}))
    # Reproduce the observed excess through float32 size/center arithmetic;
    # this is an arithmetic witness, not recovery of hidden per-mesh coordinates.
    top=f32(511.48992919921875);size=f32(f32(1080)-top);center=f32(top+f32(size/2))
    witness=dict(top=top,size=size,center=center,reconstructed_bottom=float(center)+float(size)/2)
    if witness['reconstructed_bottom']!=1080.0000305175781:raise ValueError('Arithmetic witness changed')
    paths=[pp,cp,bp,source,dp,Path(__file__).resolve(),Path(f['source_receipt'])]
    prior.frozen(OUT/'boundary-validation.json',dict(status='diagnostic_arithmetic_normalization_verified_not_integrated',records=records,witness=witness,
        bounds=dict(x=ulp32(1920),y=ulp32(1080)),
        derivation='For float32 screen endpoints bounded by dimension D, width subtraction error <=0.5 ULP(D); half-width division is exact here; float32 center addition error <=0.5 ULP(D). Reconstructed boundary error <=1 ULP(D). Applies only to verified bounded size/center arithmetic, not geometry errors or arbitrary inputs.',
        limitations='Per-mesh intermediate values were not recorded; witness supports arithmetic cause but does not uniquely reconstruct A16 internal history. Historical strict gate remains failed; normalizer is diagnostic-only.',
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(witness);print('bounds',ulp32(1920),ulp32(1080))

if __name__=='__main__':main()
