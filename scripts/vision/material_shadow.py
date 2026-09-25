"""Read-only Gazebo RGB sidecar. No flight, planner, arming or model promotion API."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time
from scripts.vision import material_routed_contrast_control as reference
from scripts.vision.locked_cpu_threads import locked_threads
from src.ml.artifacts import file_sha256
from src.sensors.camera_decoder import decode_camera_payload
from src.sensors.gazebo_camera import GazeboCameraSource
from src.vision.canonical.plan import write_record

PROTOCOL=dict(device='cpu',imgsz=640,conf=.37,iou=.7,max_det=300,agnostic_nms=False,rect=False,verbose=False)

def model_identity(seed):
    if seed not in (7,17,27):raise ValueError('Unregistered seed')
    key=f'material-routed-480-{seed}'
    cp=reference.OUT/'training'/key/'completion.json';c=reference.checked(cp)
    reference.checked(reference.OUT/'audit-v1/completion.json')
    if c['optimizer_steps']!=480:raise ValueError('Incomplete training')
    return dict(key=key,weights=c['weights'],weights_sha256=file_sha256(c['weights']),receipt=str(cp.resolve()),receipt_sha256=file_sha256(cp),
                selection='fixed seed for integration, not best-seed selection',protocol=PROTOCOL,control_authority='none',training_admitted=False,promotable=False)

def infer(model,rgb):
    import numpy as np
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected RGB8')
    started=time.monotonic()
    results=model.predict(source=np.ascontiguousarray(rgb[:,:,::-1]),**PROTOCOL)
    rows=[]
    for r in results:
        if r.boxes is None:continue
        for cls,conf,box in zip(r.boxes.cls.tolist(),r.boxes.conf.tolist(),r.boxes.xyxy.tolist()):
            rows.append(dict(class_name=r.names[int(cls)],confidence=conf,xyxy=box))
    return dict(predictions=rows,inference_call_ms=(time.monotonic()-started)*1000,control_authority='none')

def latest_put(queue,event):
    dropped=0
    if queue.full():queue.get_nowait();dropped=1
    queue.put_nowait(event)
    return dropped

def warmup(model):
    """Initialize inference before subscribing; synthetic outputs are discarded."""
    import numpy as np
    started=time.monotonic()
    timings=[infer(model,np.zeros((640,640,3),dtype=np.uint8))['inference_call_ms'] for _ in range(3)]
    finished=time.monotonic()
    return dict(status='inference_warmup_complete_before_subscription',synthetic_calls=3,call_ms=timings,
                duration_ms=(finished-started)*1000,finished_monotonic=finished,
                synthetic_predictions_discarded=True,training_admitted=False,promotable=False)

async def observe(model,out,topic,seconds,source_factory=GazeboCameraSource):
    source=source_factory(out/'payloads',topic=topic,encoder_workers=1)
    queue=asyncio.Queue(maxsize=1);counts=dict(received=0,invalid=0,dropped=0,stale=0,inferred=0)
    async def receive():
        async for event in source.events(timeout_s=5):
            counts['received']+=1
            if not event.valid:counts['invalid']+=1;continue
            counts['dropped']+=latest_put(queue,(event,time.monotonic()))
    task=None
    try:
        await asyncio.wait_for(source.start(),timeout=10)
        task=asyncio.create_task(receive());end=time.monotonic()+seconds
        with (out/'detections.jsonl').open('x') as log:
            while time.monotonic()<end:
                if task.done():task.result();raise RuntimeError('RGB stream ended')
                try:event,enqueued=await asyncio.wait_for(queue.get(),timeout=min(1,max(.001,end-time.monotonic())))
                except asyncio.TimeoutError:continue
                decode_start=time.monotonic()
                if hasattr(event,'materialize'):
                    if decode_start-event.received>1:counts['stale']+=1;continue
                    try:event=await asyncio.to_thread(event.materialize)
                    except (ValueError,TypeError):counts['invalid']+=1;continue
                frame=event.frame;age=decode_start-frame.receive_monotonic_timestamp
                if age>1:counts['stale']+=1;continue
                decoded=getattr(event,'decoded',None)
                if decoded is None:decoded=decode_camera_payload(frame,out/'payloads')
                decode_end=time.monotonic()
                result=await asyncio.to_thread(infer,model,decoded.image.array)
                decision_end=time.monotonic()
                if getattr(event,'decoded',None) is not None:
                    from src.sensors.gazebo_camera_memory import persist_event
                    await asyncio.to_thread(persist_event,event,out/'payloads')
                evidence_end=time.monotonic()
                result.update(frame=asdict(frame),rgb_sha256=decoded.image.decoded_content_sha256,receive_age_at_inference_s=age,
                              source_encode_dispatch_ms=(enqueued-frame.receive_monotonic_timestamp)*1000,
                              application_queue_wait_ms=(decode_start-enqueued)*1000,
                              decode_ms=(decode_end-decode_start)*1000,
                              inference_dispatch_and_call_ms=(decision_end-decode_end)*1000,
                              receive_to_result_ms=(decision_end-frame.receive_monotonic_timestamp)*1000,
                              result_monotonic=decision_end,
                              evidence_persist_ms=(evidence_end-decision_end)*1000,
                              receive_to_logged_result_ms=(evidence_end-frame.receive_monotonic_timestamp)*1000,
                              capture_to_inference_age_s=None,clock_limit='Simulation capture clock is not wall monotonic; transport backlog is not certified by receive age.')
                log.write(json.dumps(result,allow_nan=False)+'\n');log.flush();counts['inferred']+=1
        if counts['inferred']==0:raise RuntimeError('No valid inference frame')
        return counts
    finally:
        if task:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
        await source.stop()

def run(args):
    identity=model_identity(args.seed)
    if args.mode=='preflight':print(json.dumps(identity,indent=2));return
    if not 0<args.seconds<=120:raise ValueError('Duration must be in (0,120] seconds')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    write_record(out/'model.json',dict(**identity,inputs={identity['receipt']:identity['receipt_sha256'],str(Path(identity['weights']).resolve()):identity['weights_sha256'],str(Path(__file__).resolve()):file_sha256(__file__)}))
    try:
        from ultralytics import YOLO
        with locked_threads(4):
            model=YOLO(identity['weights'])
            if args.mode=='smoke':
                import numpy as np
                from PIL import Image
                p=reference.checked(reference.OUT/'protocol.json');row=p['pool_rows'][0]
                if file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Image changed')
                result=infer(model,np.array(Image.open(row['image_path']).convert('RGB')))
                write_record(out/'smoke.json',dict(**result,status='offline_inference_only_not_flight_validation',image=row['image_path'],inputs={row['image_path']:row['image_sha256']}))
                counts=dict(inferred=1)
            else:
                write_record(out/'warmup.json',warmup(model))
                counts=asyncio.run(observe(model,out,args.topic,args.seconds))
        write_record(out/'completion.json',dict(status=args.mode+'_complete_not_flight_certified',counts=counts,control_authority='none',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in out.rglob('*') if p.is_file()}))
    except BaseException as e:
        write_record(out/'failure.json',dict(status='sidecar_failed_no_flight_action',error=f'{type(e).__name__}: {e}',control_authority='none',training_admitted=False,promotable=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','smoke','live'),default='preflight')
    p.add_argument('--seed',type=int,choices=(7,17,27),default=7)
    p.add_argument('--topic',default='auto');p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--output',default='data/research/material-shadow-v1/run-001')
    run(p.parse_args())
