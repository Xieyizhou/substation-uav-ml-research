"""Summarize paired-order runtime measurements without safety certification."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
from scripts.vision import material_shadow as s
from src.vision.replay.static_runtime import timing_summary
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

ROOT=Path('data/research/material-shadow-v1/memory-benchmark-v1').resolve()

def main():
    s.reference.checked(ROOT/'completion.json');paths=[ROOT/'completion.json'];runs=[]
    for folder in sorted(ROOT.glob('[0-9][0-9]-*')):
        kind=folder.name.split('-')[1];p=folder/'live';c=s.reference.checked(p/'completion.json')
        if kind=='memory':s.reference.checked(p/'memory-implementation.json');paths.append(p/'memory-implementation.json')
        log=p/'detections.jsonl';rows=[json.loads(x) for x in log.read_text().splitlines()]
        if len(rows)!=c['counts']['inferred']:raise ValueError('Missing prediction rows')
        runs.append(dict(name=folder.name,pipeline=kind,counts=c['counts'],warm={k:timing_summary([r[k] for r in rows[1:]]) for k in ('source_encode_dispatch_ms','decode_ms','inference_call_ms','receive_to_result_ms','evidence_persist_ms','receive_to_logged_result_ms')},cold_live_inference_ms=rows[0]['inference_call_ms'],payload_bytes=sum(x.stat().st_size for x in (p/'payloads').rglob('*') if x.is_file()),unique_payload_files=len(list((p/'payloads/frames').glob('*')))))
        paths.extend([p/'completion.json',log])
    if len(runs)!=6:raise ValueError('Incomplete benchmark')
    groups={k:[r for r in runs if r['pipeline']==k] for k in ('png','memory')}
    medians={k:statistics.median(r['warm']['receive_to_logged_result_ms']['p50_ms'] for r in rows) for k,rows in groups.items()}
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_gazebo_camera_memory','tests.test_material_shadow','tests.test_gazebo_camera_pipeline','tests.test_camera_decoder','-q'],capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stderr)
    offline=Path('data/research/material-shadow-v1/memory-offline-v1/completion.json').resolve();s.reference.checked(offline)
    paths.extend([offline,Path(__file__).resolve(),Path('scripts/vision/verify_material_memory.py').resolve(),Path('tests/test_gazebo_camera_memory.py').resolve()])
    value=dict(status='memory_optimization_benchmark_complete',runs=runs,median_of_run_medians_ms=medians,relative_reduction=1-medians['memory']/medians['png'],offline_equivalent_frames=96,tests=tests.stdout+tests.stderr,
        limits=['Single static known-development pose, seed7, three runs per path; no flight/PX4 or significance claim.','Post-receive latency excludes transport backlog and sensor capture.','Memory path changes evidence I/O: only consumed images are stored as deduplicated lossless raw RGB; dropped frames retain counts, not images.','Raw payloads can use much more disk on changing scenes; short bounded runs only.'],training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths})
    write_record(ROOT/'report.json',value)
    lines=['# 内存RGB旁路优化测速','', '同一权重、阈值、方形填充、预热和CPU4线程；按PNG、内存、内存、PNG、PNG、内存顺序各运行15秒。无飞行或PX4。96张开发图的RGB及预测核验通过，25项相关测试通过。','', '| 轮次 | 接收/推理/丢弃 | 接收到可记录结果中位数ms | P95 ms | 推理中位数ms |','|---|---|---:|---:|---:|']
    for r in runs:
        c=r['counts'];t=r['warm'];lines.append(f"| {r['name']} | {c['received']}/{c['inferred']}/{c['dropped']} | {t['receive_to_logged_result_ms']['p50_ms']:.1f} | {t['receive_to_logged_result_ms']['p95_ms']:.1f} | {t['inference_call_ms']['p50_ms']:.1f} |")
    lines += ['',f"三轮中位数的中位数：PNG {medians['png']:.1f}ms，内存 {medians['memory']:.1f}ms，描述性降低 {value['relative_reduction']:.1%}。这是本机短时静态测试，不是跨场景速度承诺或安全认证。",'', '内存路径删除推理前PNG编码及回读解码，仍验证RGB身份并保存已推理图像的无损raw证据。保存耗时计入表中延迟。源图重复时按内容哈希复用存储；丢弃帧不保存图像，保留数量，故这不是完全相同磁盘工作量的纯算子对照。动态场景raw空间可能明显增大。', '', '接收前的Gazebo/JSON管道积压仍未知，不能宣称曝光到决策延迟已达标。模型自身的材质漏检没有因本次优化得到改善。不改变正式模型，不赋予飞控权限。', '', '入口：`.venv/bin/python -m scripts.vision.material_shadow_memory --mode live --seconds 30 --output data/research/material-shadow-v1/memory-run-001`。输出目录必须不存在。原PNG入口保留用于回归。', '', '所有产物 training_admitted=false、promotable=false。']
    report=ROOT/'report-zh.md';report.write_text('\n'.join(lines)+'\n')
    write_record(ROOT/'report-receipt.json',dict(status='report_complete',training_admitted=False,promotable=False,inputs={str(q):file_sha256(q) for q in (ROOT/'report.json',report)}))
    print(json.dumps(value['median_of_run_medians_ms']));print(value['relative_reduction'])

if __name__=='__main__':main()
