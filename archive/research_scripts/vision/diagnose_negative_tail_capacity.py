"""No-gradient reciprocal BN-statistics test for the failed negative tail."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import torch
from ultralytics import YOLO
from src.vision.canonical.plan import read_record, write_record
from src.ml.artifacts import file_sha256
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs, predict
from scripts.vision.exposure_metrics import score, summary
from scripts.vision.diagnose_bn_statistics import exchange
from scripts.vision.prepare_negative_rehearsal_control import OUT as FAILED, SOURCE

OUT = FAILED.parent / 'negative-tail-capacity-diagnosis-v1'

def main():
    OUT.mkdir(exist_ok=True)
    paired, negatives, inputs = _load_inputs()
    def forbidden(*a, **kw): raise AssertionError('Training forbidden')
    with ExitStack() as stack:
        for obj, name in ((torch.optim.Optimizer, '__init__'), (torch.Tensor, 'backward'), (YOLO, 'train'), (YOLO, 'val')):
            stack.enter_context(patch.object(obj, name, forbidden))
        torch.set_num_threads(4)
        for seed in (7,17,27):
            dest = OUT / f'{seed}.json'
            if dest.exists(): continue
            paths = [SOURCE/'training'/f'visibility-452-{seed}'/'completion.json', FAILED/'training'/f'negative-rehearsal-480-{seed}'/'completion.json']
            receipts = [read_record(p) for p in paths]
            for c in receipts:
                if file_sha256(c['weights']) != c['weights_sha256']: raise ValueError('Weight drift')
            outputs = {}; parameter_count = None
            for name, receiver, donor in [('before',0,None),('after',1,None),('after_before_stats',1,0),('before_after_stats',0,1)]:
                model = YOLO(receipts[receiver]['weights'])
                parameter_count = sum(p.numel() for p in model.model.parameters())
                changed = []
                if donor is not None:
                    other = YOLO(receipts[donor]['weights']); changed = exchange(model.model, other.model); del other
                rows = []; neg = []
                with torch.inference_mode():
                    for row, truth in paired:
                        formal = predict(model,row['image_path'],.37)
                        rows.append(score(row,truth,formal,predict(model,row['image_path'],.001)))
                    for row in negatives:
                        neg.append(predict(model,row['image_path'],.37))
                if donor is None:
                    ep = (SOURCE/'evaluation-v2'/f'visibility-452-{seed}.json') if receiver == 0 else FAILED/'evaluation-v1'/f'negative-rehearsal-480-{seed}.json'
                    saved = read_record(ep)
                    if rows != saved['rows'] or neg != [r['predictions'] for r in saved['negative_rows']]: raise ValueError('Endpoint reproduction failed')
                    paths.append(ep)
                outputs[name] = dict(rows=rows,negative_predictions=neg,summary={v:summary([r for r in rows if r['variant']==v]) for v in ('original','material','background','lighting')},fpr=sum(bool(p) for p in neg)/48,changed_buffers=changed)
                print(seed,name,outputs[name]['summary']['original']['instance_recall'],outputs[name]['fpr'],flush=True)
            paths += [Path(c['weights']) for c in receipts]+[Path(__file__).resolve()]
            write_record(dest,dict(status='reciprocal_buffer_diagnosis_complete',parameter_count=parameter_count,outputs=outputs,training_admitted=False,promotable=False,inputs={**inputs,**{str(p):file_sha256(p) for p in paths}}))

if __name__ == '__main__': main()
