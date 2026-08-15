const $=s=>document.querySelector(s);let recordings=[],scenarios=[],page=1,operatorToken='',lidarState=null,acceptanceState=null,preflightState=null,profileState=null;
const api=async path=>{const r=await fetch(path,{cache:'no-store'});const v=await r.json();if(!r.ok)throw new Error(v.error||r.statusText);return v};
const post=async(path,value)=>{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Sandbox-Token':operatorToken},body:JSON.stringify(value)});const v=await r.json();if(!r.ok)throw new Error(v.error||r.statusText);return v};
const esc=v=>String(v??'—').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
function activateTab(name){document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));const tab=document.querySelector(`.tab[data-tab="${name}"]`);tab?.classList.add('active');$('#'+name)?.classList.add('active')}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>activateTab(b.dataset.tab));

async function refresh(){
  const [profile,version,setup,dash,runtime,doctor,recs,available,operator,research,experiments,lidar,acceptance,preflight]=await Promise.all([api('/api/profile'),api('/api/version'),api('/api/setup'),api('/api/dashboard'),api('/api/runtime'),api('/api/doctor'),api('/api/recordings'),api('/api/scenarios'),api('/api/operator'),api('/api/research'),api('/api/experiments'),api('/api/lidar'),api('/api/acceptance'),api('/api/preflight')]);
  recordings=recs;scenarios=available;operatorToken=operator.operator_token;renderProfile(profile);renderVersion(version);renderSetup(setup);renderDashboard(dash,runtime);renderResearch(research);renderExperiments(experiments);renderLidar(lidar);renderAcceptance(acceptance);renderPreflight(preflight);renderDoctor(doctor);renderSelectors();renderOperator(operator);
  $('#updated').textContent=`Observed ${new Date().toLocaleTimeString()}`;
}
const pct=v=>v==null?'—':`${(Number(v)*100).toFixed(2)}%`;
const fixed=(v,d=2)=>v==null?'—':Number(v).toFixed(d);
function renderProfile(value){profileState=value;$('#profile-badge').textContent=value.title;$('#profile-description').textContent=value.description;document.querySelectorAll('[data-full-profile]').forEach(control=>control.disabled=!value.flight_enabled);document.querySelectorAll('[data-full-section]').forEach(section=>section.hidden=!value.flight_enabled);document.querySelectorAll('#experiment-workflow option').forEach(option=>option.disabled=!value.available_workflows.includes(option.value));if(!value.available_workflows.includes($('#experiment-workflow').value))$('#experiment-workflow').value=value.available_workflows[0];updateWorkflowInputs()}
function renderVersion(value){$('#version-label').textContent=`App ${value.macos_app_version} · Sandbox ${value.sandbox_product_version}`}
function stageCard(label,value){const ok=value.status==='complete',detail=value.identity?.slice(0,12)||(ok?'verified result':'artifact unavailable');return `<div class="pipeline-stage ${ok?'ready':''}"><span class="badge ${ok?'pass':value.status==='invalid'?'failure':'warning'}">${esc(value.status)}</span><strong>${esc(label)}</strong><small>${esc(detail)}</small></div>`}
function renderResearch(r){
  $('#research-status').textContent=`${r.complete_stage_count} / ${r.stage_count} stages complete`;
  $('#research-pipeline').innerHTML=[['Training view',r.training_view],['Model package',r.model_package],['Paired blind',r.paired_blind],['Static replay',r.static_replay]].map(([label,value])=>stageCard(label,value)).join('<i aria-hidden="true">→</i>');
  const t=r.training_view;$('#training-summary').innerHTML=t.status==='complete'?`<div class="research-facts"><span>Training frames<b>${esc(t.train_frames)}</b></span><span>Selection validation<b>${esc(t.selection_validation_frames)}</b></span><span>Full validation<b>${esc(t.full_validation_frames)}</b></span><span>Sampling<b>${esc(t.algorithm)}</b></span></div><small class="artifact-path">${esc(t.path)}</small>`:`<p>Training view is ${esc(t.status)}.</p>`;
  const p=r.model_package;$('#package-summary').innerHTML=p.status==='complete'?`<div class="research-facts"><span>Architecture<b>${esc(p.architecture)}</b></span><span>Frozen threshold<b>${fixed(p.threshold)}</b></span><span>ONNX inputs<b>${esc(p.exports.join(' / '))}</b></span><span>Package identity<b>${esc(p.identity.slice(0,12))}…</b></span></div><small class="artifact-path">${esc(p.path)}</small>`:`<p>Model package is ${esc(p.status)}${p.error?`: ${esc(p.error)}`:'.'}</p>`;
  const b=r.paired_blind;if(b.status!=='complete'){$('#blind-metrics').innerHTML=`<p>Paired blind result is ${esc(b.status)}.</p>`;$('#class-recall').innerHTML='';$('#blind-comparison').innerHTML='';return}
  $('#blind-commit').textContent=`${b.frame_count.toLocaleString()} frames · commit ${b.commit.slice(0,8)}`;
  $('#blind-metrics').innerHTML=[['v2 mAP50–95',pct(b.v2.map50_95)],['v2 macro-F1',pct(b.v2.macro_f1)],['Precision',pct(b.v2.precision)],['Recall',pct(b.v2.recall)],['Small recall',pct(b.v2.small_recall)],['No-target FPR',pct(b.v2.no_target_fpr)]].map(([k,v])=>`<div class="metric"><b>${v}</b><span>${k}</span></div>`).join('');
  $('#class-recall').innerHTML=Object.entries(b.v2.per_class_recall).map(([name,value])=>`<div><span>${esc(name)}</span><b>${pct(value)}</b><div class="mini-track"><i style="width:${Number(value)*100}%"></i></div></div>`).join('');
  $('#blind-comparison').innerHTML=`<strong>v2 improved macro-F1 by ${pct(b.macro_f1_delta)}</strong><span>mAP50–95 Δ ${pct(b.map_delta)} · paired bootstrap 95% CI ${pct(b.bootstrap_ci95[0])} to ${pct(b.bootstrap_ci95[1])} · improvement probability ${pct(b.bootstrap_improvement_probability)}</span>`;
  const replay=r.static_replay;$('#replay-status').textContent=`${replay.completed} / ${replay.total} completed`;
  $('#replay-rows').innerHTML=replay.rows.map(x=>`<tr><td><b>${x.input_size}</b></td><td>${esc(x.policy)}</td><td><span class="source-tag ${x.source==='controlled_replicate'?'selected':''}">${esc(x.source)}</span></td><td>${fixed(x.throughput_fps)}</td><td>${fixed(x.p50_ms)} ms</td><td>${fixed(x.p95_ms)} ms</td><td>${pct(x.precision)}</td><td>${pct(x.recall)}</td><td>${pct(x.small_recall)}</td></tr>`).join('');
  $('#replay-note').textContent=replay.note||'';
}
function gateBadge(value){const cls=value.passed?'pass':value.status==='failed'||value.status==='invalid'?'failure':'warning';return `<span class="badge ${cls}">${esc(value.status)}</span>`}
function renderLidar(value){
  lidarState=value;const replay=value.replay,closed=value.closed_loop;
  $('#lidar-status').textContent=value.status==='complete'?'Replay and closed-loop gates passed':'Gate work remains';
  $('#lidar-replay-badge').innerHTML=gateBadge(replay);
  $('#lidar-replay-summary').innerHTML=replay.status==='missing'?'<p>No replay receipt is available.</p>':`<div class="research-facts"><span>Model<b>${esc(replay.model_id)}</b></span><span>Macro-F1<b>${pct(replay.macro_f1)}</b></span><span>Danger recall<b>${pct(replay.danger_recall)}</b></span><span>CPU P95<b>${fixed(replay.inference_p95_ms,3)} ms</b></span></div><small class="artifact-path">${esc(replay.path||replay.error)}</small>`;
  $('#lidar-closed-badge').innerHTML=gateBadge(closed);
  const total=closed.total||0,progress=total?100*closed.completed/total:0;
  $('#lidar-closed-summary').innerHTML=`<div class="progress-row"><strong>${closed.completed||0} / ${total}</strong><span>${closed.scenario_count||0} scenarios</span></div><div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div><div class="research-facts gate-facts"><span>Missions / landings<b>${closed.mission_success||0} / ${closed.landing_success||0}</b></span><span>Collisions / buffer entries<b>${closed.collision_count||0} / ${closed.buffer_entry_count||0}</b></span><span>Minimum sensor health<b>${pct(closed.sensor_health_min)}</b></span><span>Maximum inference P95<b>${fixed(closed.inference_p95_max_ms,3)} ms</b></span></div><small class="artifact-path">${esc(closed.study_id||closed.error||'No closed-loop study')}</small>`;
  $('#lidar-replay-start').disabled=!profileState?.flight_enabled;
  $('#lidar-closed-start').disabled=!closed.pending||!profileState?.flight_enabled;
  $('#lidar-closed-option').disabled=!closed.pending||!profileState?.flight_enabled;
  $('#experiment-workflow option[value="lidar_closed_loop"]').disabled=!closed.pending||!profileState?.available_workflows.includes('lidar_closed_loop');
}
function defaultExperimentName(){const d=new Date(),part=n=>String(n).padStart(2,'0');return `validation-416-${d.getFullYear()}${part(d.getMonth()+1)}${part(d.getDate())}-${part(d.getHours())}${part(d.getMinutes())}${part(d.getSeconds())}`}
function renderExperiments(rows){
  if(!$('#experiment-name').value)$('#experiment-name').value=defaultExperimentName();
  $('#experiment-history').innerHTML=rows.length?rows.map(x=>{const complete=x.state==='complete',m=x.metrics||{},timing=x.timing||{},resources=x.resources||{},schedule=x.frame_skip_interval===1?'every frame':`every ${x.frame_skip_interval} frames`;return `<article class="experiment-card"><header><span class="badge ${complete?'pass':x.state==='failed'||x.state==='invalid'?'failure':'warning'}">${esc(x.state)}</span><div><h3>${esc(x.experiment_id)}</h3><small>${esc(x.partition||'unavailable')} · ${esc(x.input_size||'—')} px · ${esc(schedule)}</small></div></header>${complete?`<div class="experiment-facts"><span>Precision<b>${pct(m.precision)}</b></span><span>Recall<b>${pct(m.recall)}</b></span><span>Small recall<b>${pct(m.small_object_recall)}</b></span><span>No-target FPR<b>${pct(m.no_target_false_positive_rate)}</b></span><span>P95 latency<b>${fixed(timing.p95_ms)} ms</b></span><span>Throughput<b>${fixed(resources.throughput_fps)} FPS</b></span></div>`:`<div class="experiment-facts"><span>Source frames<b>${esc(x.source_frame_count||'—')}</b></span><span>Inference frames<b>${esc(x.inference_frame_count||'—')}</b></span><span>Threshold<b>${fixed(x.confidence_threshold)}</b></span></div>`}${x.error?`<p class="experiment-error">${esc(x.error)}</p>`:''}</article>`}).join(''):'<article class="experiment-empty">No sandbox evaluation recipes have run yet.</article>';
}
function renderAcceptance(value){
  acceptanceState=value;const gate=value.acceptance,labels={visual_replay:'Visual replay',lidar_replay:'LiDAR replay',capability_challenge:'Capability challenge',closed_loop_flight:'Closed-loop flight',safe_process_stop:'Safe process stop',failure_diagnostics:'Failure diagnostics'};
  $('#acceptance-badge').innerHTML=gateBadge(gate);
  $('#acceptance-checks').innerHTML=Object.entries(gate.checks||{}).map(([name,check])=>`<div class="acceptance-check"><span class="badge ${check.passed?'pass':'warning'}">${check.passed?'pass':'pending'}</span><b>${esc(labels[name]||name)}</b><small>${esc(check.identity?.slice(0,12)||check.error||'evidence available after gate run')}</small></div>`).join('');
  $('#acceptance-note').textContent=gate.scope_note||'The acceptance gate pairs visual and LiDAR evidence; it does not claim synchronized sensor fusion.';
  $('#workflow-history').innerHTML=value.workflows.length?value.workflows.map(row=>`<article class="experiment-card"><header><span class="badge ${row.state==='complete'?'pass':row.state==='invalid'||row.state==='failed'?'failure':'warning'}">${esc(row.state)}</span><div><h3>${esc(row.workflow||row.job_id)}</h3><small>${esc(row.action||row.error)} · ${esc(row.ended_at||'not finished')}</small></div></header>${row.receipt_identity_sha256?`<small class="artifact-path">receipt ${esc(row.receipt_identity_sha256.slice(0,16))}…</small>`:''}</article>`).join(''):'<article class="experiment-empty">New managed runs will produce identity-bound workflow receipts.</article>';
}
function renderPreflight(value){
  preflightState=value;$('#preflight-badge').innerHTML=`<span class="badge ${value.ready_for_large_lidar?'pass':'warning'}">${esc(value.status)}</span>`;
  const labels={doctor:'Environment doctor',disk:'Disk capacity',runtime_idle:'Runtime idle',capability_challenge:'Capability challenge'};
  $('#preflight-checks').innerHTML=Object.entries(value.checks).map(([name,check])=>`<div class="acceptance-check"><span class="badge ${check.passed?'pass':'warning'}">${check.passed?'pass':'blocked'}</span><b>${esc(labels[name]||name)}</b><small>${esc(check.detail)}</small></div>`).join('');
  const challenge=value.challenge||{};$('#challenge-summary').innerHTML=challenge.status==='missing'?'<p>No capability receipt is available.</p>':`<div class="research-facts"><span>Status<b>${esc(challenge.status)}</b></span><span>Model<b>${esc(challenge.model_id)}</b></span><span>Study<b>${esc(challenge.study_id)}</b></span><span>Receipt<b>${esc(challenge.challenge_receipt_identity_sha256?.slice(0,12)||'unavailable')}</b></span></div><small class="artifact-path">${esc(challenge.path||challenge.error)}</small>`;
  $('#challenge-start').disabled=!profileState?.flight_enabled;$('#challenge-stop').disabled=true;
}
function renderDashboard(d,runtime){
  $('#progress').innerHTML=`<div class="progress-row"><div><span class="kicker">OVERALL PROGRESS</span><br><strong>${d.completed} / ${d.total}</strong></div><div>${d.remaining} scenarios remaining</div></div><div class="progress-track"><div class="progress-fill" style="width:${d.progress_percent}%"></div></div>`;
  $('#counts').innerHTML=Object.entries(d.counts).map(([k,v])=>`<div class="metric"><b>${v}</b><span>${esc(k)}</span></div>`).join('');
  const scenario=d.current||d.next;
  $('#scenario').innerHTML=scenario?Object.entries({Status:scenario.state,Scenario:scenario.scenario_id,Role:scenario.dataset_role,Layout:scenario.layout,Route:scenario.route,Seed:scenario.seed,Target:scenario.target_class}).map(([k,v])=>`<div class="scenario-row"><span>${k}</span><strong>${esc(v)}</strong></div>`).join(''):'<p>All scenarios are complete.</p>';
  $('#runtime').innerHTML=runtime.map(x=>`<div class="status-row"><span><i class="dot ${x.alive?'alive':''}"></i> ${esc(x.name)}</span><span>${!x.available?'unavailable':x.alive?`PID ${x.pid}`:'not detected'}</span></div>`).join('');
  $('#roles').innerHTML=Object.entries(d.role_counts).map(([k,v])=>`<span class="role"><b>${v}</b> ${esc(k)}</span>`).join('');
}
function renderDoctor(checks){
  $('#checks').innerHTML=checks.map(x=>`<div class="check"><span class="badge ${x.status}">${esc(x.status)}</span><strong>${esc(x.name)}</strong><div>${esc(x.explanation)}${x.action?`<br><small>${esc(x.action)}</small>`:''}</div></div>`).join('');
}
function renderSelectors(){
  const options=recordings.map(x=>`<option value="${esc(x.recording_id)}">${esc(x.scenario_id)} · ${esc(x.role)}</option>`).join('');
  $('#recording').innerHTML=options;
  $('#log-scenario').innerHTML=recordings.map(x=>`<option value="${esc(x.scenario_id)}">${esc(x.scenario_id)}</option>`).join('');
  $('#load-frames').disabled=!recordings.length;$('#load-log').disabled=!recordings.length;
  if(!recordings.length){$('#recording-progress').innerHTML='<div class="summary">No non-blind recordings are present in the configured collection root.</div>';$('#log-output').textContent='No non-blind recorded scenarios are present.'}
}

function renderOperator(value){
  const active=value.active_job;
  $('#operator-state').innerHTML=active?`<span class="badge warning">${esc(active.state)}</span><strong>${esc(active.action)}</strong><span>${esc(active.job_id)}</span>`:'<span class="badge pass">idle</span><strong>Ready for one managed job</strong>';
  $('#operator-start').disabled=Boolean(active);$('#operator-stop').disabled=!active;$('#operator-stop').dataset.job=active?.job_id||'';
  $('#job-history').innerHTML=value.history.length?value.history.map(job=>`<button class="job-row" data-job="${esc(job.job_id)}" ${job.sensitive?'data-sensitive="true"':''}><span><b>${esc(job.action)}</b><small>${esc(job.created_at)}</small></span><span class="badge ${job.state==='complete'?'pass':job.state==='failed'?'failure':'warning'}">${esc(job.state)}</span></button>`).join(''):'<p>No managed jobs have run.</p>';
  document.querySelectorAll('.job-row').forEach(row=>row.onclick=()=>loadJobLog(row.dataset.job,row.dataset.sensitive==='true'));
  $('#operator-scenario').innerHTML=scenarios.map(row=>`<option value="${esc(row.scenario_id)}">${esc(row.scenario_id)} · ${esc(row.dataset_role)}</option>`).join('');
  $('#experiment-start').disabled=Boolean(active);
  $('#lidar-replay-start').disabled=Boolean(active)||!profileState?.flight_enabled;$('#lidar-closed-start').disabled=Boolean(active)||!lidarState?.closed_loop?.pending||!preflightState?.ready_for_large_lidar||!profileState?.flight_enabled;$('#lidar-stop').disabled=!active;$('#lidar-stop').dataset.job=active?.job_id||'';
  $('#challenge-start').disabled=Boolean(active)||!profileState?.flight_enabled;$('#challenge-stop').disabled=!active;$('#challenge-stop').dataset.job=active?.job_id||'';
  updateOperatorInputs();
}
function updateOperatorInputs(){const smoke=$('#operator-action').value==='flight-smoke';$('#operator-scenario').disabled=!smoke;$('#operator-scenario').hidden=!smoke}
async function refreshOperator(){try{const [value,experiments,lidar,acceptance,preflight]=await Promise.all([api('/api/operator'),api('/api/experiments'),api('/api/lidar'),api('/api/acceptance'),api('/api/preflight')]);operatorToken=value.operator_token;renderExperiments(experiments);renderLidar(lidar);renderAcceptance(acceptance);renderPreflight(preflight);renderOperator(value)}catch(e){$('#operator-state').textContent=e.message}}
async function loadJobLog(job,sensitive){if(sensitive){$('#job-log').textContent='Blind collection job logs are sealed.';return}try{$('#job-log').innerHTML=(await api(`/api/operator/log/${encodeURIComponent(job)}?limit=300`)).join('\n')||'Log is empty.'}catch(e){$('#job-log').textContent=e.message}}
$('#operator-action').onchange=updateOperatorInputs;
let pendingConfirmation=null,confirmationTrigger=null;
function actionError(message){const status=$('#action-status');status.textContent=message;status.classList.add('visible');clearTimeout(actionError.timer);actionError.timer=setTimeout(()=>status.classList.remove('visible'),8000)}
function requestConfirmation({title,detail,facts={},confirmLabel='Confirm',danger=false}){
  const dialog=$('#action-dialog');
  if(pendingConfirmation){pendingConfirmation(false);pendingConfirmation=null}
  if(dialog.open)dialog.close('cancel');
  confirmationTrigger=document.activeElement;
  $('#action-dialog-title').textContent=title;$('#action-dialog-detail').textContent=detail;
  $('#action-dialog-facts').innerHTML=Object.entries(facts).map(([name,value])=>`<dt>${esc(name)}</dt><dd>${esc(value)}</dd>`).join('');
  const confirmButton=$('#action-dialog-confirm');confirmButton.textContent=confirmLabel;confirmButton.classList.toggle('danger',danger);
  dialog.showModal();confirmButton.focus();
  return new Promise(resolve=>{pendingConfirmation=resolve});
}
$('#action-dialog').addEventListener('close',()=>{const resolve=pendingConfirmation;pendingConfirmation=null;if(resolve)resolve($('#action-dialog').returnValue==='confirm');confirmationTrigger?.focus();confirmationTrigger=null});
$('#action-dialog').addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();$('#action-dialog').close('cancel')}});
async function startManaged(action,scenario=null){const accepted=await requestConfirmation({title:`Start ${action}?`,detail:'Only one managed Sandbox job may run. Environment, output budget, and runtime conflicts will be checked before launch.',facts:{Action:action,Scenario:scenario||'fixed by workflow'},confirmLabel:'Start job'});if(!accepted)return;try{await post('/api/operator/start',{action,scenario_id:scenario});await refreshOperator()}catch(e){actionError(e.message)}}
async function stopManaged(job){if(!job)return;const accepted=await requestConfirmation({title:'Stop active job?',detail:'The Sandbox will request graceful shutdown, then clean up its managed process groups.',facts:{Job:job},confirmLabel:'Stop safely',danger:true});if(!accepted)return;try{await post('/api/operator/stop',{job_id:job});await refreshOperator()}catch(e){actionError(e.message)}}
$('#operator-start').onclick=()=>{const action=$('#operator-action').value,scenario=action==='flight-smoke'?$('#operator-scenario').value:null;startManaged(action,scenario)};
$('#operator-stop').onclick=()=>stopManaged($('#operator-stop').dataset.job);
$('#lidar-replay-start').onclick=()=>startManaged('lidar-replay-gate');$('#lidar-closed-start').onclick=()=>startManaged('lidar-closed-loop-next');$('#lidar-stop').onclick=()=>stopManaged($('#lidar-stop').dataset.job);
$('#challenge-start').onclick=()=>startManaged('lidar-challenge-gate');$('#challenge-stop').onclick=()=>stopManaged($('#challenge-stop').dataset.job);
function updateWorkflowInputs(){const workflow=$('#experiment-workflow').value,visual=workflow==='visual_replay';document.querySelectorAll('.visual-option').forEach(x=>x.hidden=!visual);const notes={demo_contract:'Trains and evaluates a tiny deterministic classifier on synthetic features. The result demonstrates workflow provenance and is never formal evidence.',visual_replay:'Visual replay uses non-blind validation data. Model package, threshold and membership cannot be overridden.',lidar_replay:'LiDAR replay revalidates the latest passed candidate with its fixed model and dataset identities.',lidar_closed_loop:'Closed-loop runs exactly one pending approved flight and requires a clean simulator runtime.',multimodal_acceptance:'Sandbox v1 acceptance binds existing visual, LiDAR and closed-loop evidence and runs offline supervisor lifecycle checks.'};$('#workflow-note').textContent=notes[workflow];}
$('#experiment-workflow').onchange=updateWorkflowInputs;updateWorkflowInputs();
$('#experiment-start').onclick=async()=>{const workflow=$('#experiment-workflow').value,parameters={workflow};if(workflow==='visual_replay'){const limit=$('#experiment-limit').value;Object.assign(parameters,{name:$('#experiment-name').value.trim(),partition:$('#experiment-partition').value,input_size:Number($('#experiment-size').value),frame_skip_interval:Number($('#experiment-skip').value),frame_limit:limit==='all'?null:Number(limit)})}const accepted=await requestConfirmation({title:`Run ${workflow}?`,detail:'This creates an identity-bound recipe and runs only fixed, allow-listed inputs.',facts:{Workflow:workflow,Partition:parameters.partition||'fixed by workflow'},confirmLabel:'Create and run'});if(!accepted)return;try{await post('/api/operator/start',{action:'workflow-run',parameters});if(workflow==='visual_replay')$('#experiment-name').value='';await refreshOperator()}catch(e){actionError(e.message)}};

$('#load-log').onclick=async()=>{
  const scenario=encodeURIComponent($('#log-scenario').value),kind=encodeURIComponent($('#log-kind').value);
  try{const lines=await api(`/api/logs/${scenario}/${kind}?limit=200`);$('#log-output').innerHTML=lines.length?lines.map(x=>`<span class="log-${x.level}">${x.number.toString().padStart(5)}  ${x.text}</span>`).join('\n'):'Log file is not present.'}catch(e){$('#log-output').textContent=e.message}
};
$('#load-frames').onclick=()=>{page=1;loadFrames()};
async function loadFrames(){
  const id=$('#recording').value;if(!id)return;
  try{const [data,progress]=await Promise.all([api(`/api/frames/${encodeURIComponent(id)}?page=${page}&page_size=24`),api(`/api/progress/${encodeURIComponent(id)}`)]);renderProgress(progress);renderFrames(data)}catch(e){$('#frame-grid').textContent=e.message}
}
function renderProgress(p){
  const truths=Object.entries(p.truth_counts).map(([k,v])=>`<span class="truth"><b>${v}</b> ${esc(k)}</span>`).join('');
  const phases=p.phases.map(x=>`<div class="timeline-item"><i></i><span><b>${esc(x.phase)}</b><small>${x.frames} frames · ${x.start.toFixed(3)}–${x.end.toFixed(3)}s</small></span></div>`).join('');
  const events=p.events.map(x=>`<div class="timeline-item"><i></i><span><b>${esc(x.event)}</b><small>${x.simulation_timestamp==null?'time unavailable':Number(x.simulation_timestamp).toFixed(3)+'s'} · ${esc(x.phase)}</small></span></div>`).join('');
  $('#recording-progress').innerHTML=`<div class="summary"><b>${p.frame_count}</b> frames · ${p.elapsed_simulation_time==null?'elapsed unavailable':p.elapsed_simulation_time.toFixed(2)+' simulation seconds'} · target ${esc(p.target_class)} · receipt ${esc(p.validation_receipt)}<br>${truths}</div><div class="timeline"><div><h3>Route phases</h3>${phases||'<p>No phase annotations.</p>'}</div><div><h3>Mission events</h3>${events||'<p>No mission events.</p>'}</div></div>`;
}
function renderFrames(d){
  $('#frame-grid').innerHTML=d.frames.map((f,i)=>`<button class="frame" data-index="${i}"><img loading="lazy" src="${f.image_url}" alt="Frame ${esc(f.sequence)}"><div><b>#${f.sequence} · ${esc(f.mission_phase)}</b><small>${f.simulation_timestamp.toFixed(3)}s · ${esc(f.synchronization_status)}</small><small>${esc(f.annotation_summary)}</small></div></button>`).join('');
  document.querySelectorAll('.frame').forEach(b=>b.onclick=()=>showFrame(d.frames[Number(b.dataset.index)]));
  const pages=Math.max(1,Math.ceil(d.total_frames/d.page_size));$('#pager').innerHTML=`<button id="prev" ${page<=1?'disabled':''}>Previous</button><span>Page ${page} of ${pages}</span><button id="next" ${page>=pages?'disabled':''}>Next</button>`;
  $('#prev').onclick=()=>{page--;loadFrames()};$('#next').onclick=()=>{page++;loadFrames()};
}
function showFrame(f){
  const dialog=$('#viewer'),img=$('#full-image'),canvas=$('#overlay');$('#frame-detail').textContent=`Frame ${f.sequence} · ${f.mission_phase} · ${f.annotation_summary}`;
  img.onload=()=>{canvas.width=img.clientWidth;canvas.height=img.clientHeight;const c=canvas.getContext('2d'),sx=img.clientWidth/f.width,sy=img.clientHeight/f.height;c.lineWidth=2;c.font='13px sans-serif';f.boxes.forEach(b=>{const [x1,y1,x2,y2]=b.bbox;c.strokeStyle='#ffb448';c.fillStyle='#ffb448';c.strokeRect(x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy);c.fillText(b.class_name,x1*sx+3,y1*sy+14)})};img.src=f.image_url;dialog.showModal();
}
$('#close-viewer').onclick=()=>$('#viewer').close();$('#refresh').onclick=refresh;
refresh().catch(e=>{document.querySelector('main').textContent=`Inspector unavailable: ${e.message}`});
setInterval(refreshOperator,3000);
