"""Final report of latest-before-decode controlled static measurements."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
from scripts.vision import material_shadow as s
from src.vision.replay.static_runtime import timing_summary
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256
from src.sensors.types import CameraFrame
from src.sensors.camera_decoder import decode_camera_payload

ROOT=Path('data/research/material-shadow-v1/latest-benchmark-v1').resolve()

def main():
    s.reference.checked(ROOT/'completion.json');paths=[ROOT/'completion.json'];runs=[]
    for folder in sorted(ROOT.glob('[0-9][0-9]-*')):
        kind=folder.name.split('-')[1];p=folder/'live';c=s.reference.checked(p/'completion.json')
        if kind=='latest':s.reference.checked(p/'latest-implementation.json');paths.append(p/'latest-implementation.json')
        log=p/'detections.jsonl';rows=[json.loads(x) for x in log.read_text().splitlines()]
        if len(rows)!=c['counts']['inferred']:raise ValueError('Missing prediction rows')
        if kind=='latest':
            seen=set()
            for row in rows:
                f=row['frame']
                if f['payload_relative_path'] in seen:continue
                decoded=decode_camera_payload(CameraFrame(**f),p/'payloads')
                if decoded.image.decoded_content_sha256!=row['rgb_sha256']:raise ValueError('Saved pixels do not match inference evidence')
                seen.add(f['payload_relative_path'])
        fields=('source_encode_dispatch_ms','application_queue_wait_ms','decode_ms','inference_call_ms','receive_to_result_ms','evidence_persist_ms','receive_to_logged_result_ms')
        runs.append(dict(name=folder.name,pipeline=kind,counts=c['counts'],warm={k:timing_summary([r[k] for r in rows[1:]]) for k in fields},warm_output_hz=(len(rows)-1)/(rows[-1]['result_monotonic']-rows[0]['result_monotonic']),payload_bytes=sum(x.stat().st_size for x in (p/'payloads').rglob('*') if x.is_file())))
        paths.extend([p/'completion.json',log])
    if len(runs)!=6:raise ValueError('Incomplete benchmark')
    med={k:statistics.median(r['warm']['receive_to_logged_result_ms']['p50_ms'] for r in runs if r['pipeline']==k) for k in ('png','latest')}
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_gazebo_camera_memory','tests.test_material_shadow','tests.test_gazebo_camera_pipeline','tests.test_camera_decoder','-q'],capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stderr)
    offline=Path('data/research/material-shadow-v1/latest-offline-v1/completion.json').resolve();s.reference.checked(offline)
    paths.extend([offline,Path(__file__).resolve(),Path('scripts/vision/verify_material_latest.py').resolve(),Path('tests/test_gazebo_camera_memory.py').resolve(),Path('src/sensors/gazebo_camera_memory.py').resolve(),Path('src/sensors/gazebo_camera_latest.py').resolve()])
    value=dict(status='latest_memory_optimization_tested',runs=runs,median_of_run_medians_ms=med,relative_reduction=1-med['latest']/med['png'],offline_equivalent_frames=96,tests=tests.stdout+tests.stderr,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths})
    write_record(ROOT/'report.json',value)
    lines=['# 最新消息优先内存路径：优化与测速','', '模型seed7、640方形填充、阈值0.37、NMS 0.7、max_det 300、预热及CPU4线程保持不变。三轮PNG与三轮新路径交替测试，每轮15秒，不并行其他推理、不启动PX4或飞行。','', '先前“逐帧解析到内存”方案未加速：当时接收后延迟三轮中位数的中位数为PNG184.7ms、内存191.7ms。该负结果保留，不删除。新方案将最新帧队列移到JSON/base64解析前，避免为随后丢弃的帧做重解析工作。','', '|轮次|接收/推理/主动丢弃|接收到可记录结果中位数ms|P95 ms|推理中位数ms|','|---|---|---:|---:|---:|']
    for r in runs:
        c=r['counts'];t=r['warm'];lines.append(f"|{r['name']}|{c['received']}/{c['inferred']}/{c['dropped']}|{t['receive_to_logged_result_ms']['p50_ms']:.1f}|{t['receive_to_logged_result_ms']['p95_ms']:.1f}|{t['inference_call_ms']['p50_ms']:.1f}|")
    lines += ['',f"三轮中位数的中位数：PNG {med['png']:.1f}ms，新路径 {med['latest']:.1f}ms，描述性降低 {value['relative_reduction']:.1%}。包括已推理帧的证据保存耗时，不包含接收前传输和传感器捕获时间。",'', '96张开发图逐像素RGB及预测/匹配实例一致性核验通过；预测浮点容差坐标0.001像素、置信度0.00001。26项相关测试通过，不声明全仓通过。三轮重复静态场景不是新增独立场景，不宣称统计显著或动态飞行收益。', '', '证据策略变化：PNG参考保存所有已解析帧；新路径只保存选中推理帧的无损raw RGB，相同内容去重。主动丢弃的原始消息未解析，保留丢弃计数，不认证其像素有效性。动态图像下raw磁盘开销可能较大，仍限定短时诊断。', '', '接口为 scripts.vision.material_shadow_latest，原PNG入口继续保留。模型未训练或晋升，所有产物training_admitted=false、promotable=false。端到端捕获时间映射、移动视角、PX4并发及故障保护仍未验证，不赋予飞控权限。']
    report=ROOT/'report-zh.md';report.write_text('\n'.join(lines)+'\n')
    write_record(ROOT/'report-receipt.json',dict(status='report_complete',training_admitted=False,promotable=False,inputs={str(q):file_sha256(q) for q in (ROOT/'report.json',report)}))
    print(json.dumps(med));print(value['relative_reduction'])

if __name__=='__main__':main()
