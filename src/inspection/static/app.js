const $=s=>document.querySelector(s);let recordings=[],scenarios=[],page=1,operatorToken='';
const api=async path=>{const r=await fetch(path,{cache:'no-store'});const v=await r.json();if(!r.ok)throw new Error(v.error||r.statusText);return v};
const post=async(path,value)=>{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Sandbox-Token':operatorToken},body:JSON.stringify(value)});const v=await r.json();if(!r.ok)throw new Error(v.error||r.statusText);return v};
const esc=v=>String(v??'—').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');$('#'+b.dataset.tab).classList.add('active')});

async function refresh(){
  const [dash,runtime,doctor,recs,available,operator]=await Promise.all([api('/api/dashboard'),api('/api/runtime'),api('/api/doctor'),api('/api/recordings'),api('/api/scenarios'),api('/api/operator')]);
  recordings=recs;scenarios=available;operatorToken=operator.operator_token;renderDashboard(dash,runtime);renderDoctor(doctor);renderSelectors();renderOperator(operator);
  $('#updated').textContent=`Observed ${new Date().toLocaleTimeString()}`;
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
  updateOperatorInputs();
}
function updateOperatorInputs(){const smoke=$('#operator-action').value==='flight-smoke';$('#operator-scenario').disabled=!smoke;$('#operator-scenario').hidden=!smoke}
async function refreshOperator(){try{const value=await api('/api/operator');operatorToken=value.operator_token;renderOperator(value)}catch(e){$('#operator-state').textContent=e.message}}
async function loadJobLog(job,sensitive){if(sensitive){$('#job-log').textContent='Blind collection job logs are sealed.';return}try{$('#job-log').innerHTML=(await api(`/api/operator/log/${encodeURIComponent(job)}?limit=300`)).join('\n')||'Log is empty.'}catch(e){$('#job-log').textContent=e.message}}
$('#operator-action').onchange=updateOperatorInputs;
$('#operator-start').onclick=async()=>{const action=$('#operator-action').value,scenario=action==='flight-smoke'?$('#operator-scenario').value:null;if(!confirm(`Start ${action}? Only one sandbox job may run.`))return;try{await post('/api/operator/start',{action,scenario_id:scenario});await refreshOperator()}catch(e){alert(e.message)}};
$('#operator-stop').onclick=async()=>{const job=$('#operator-stop').dataset.job;if(!job||!confirm('Stop this job and its managed process groups?'))return;try{await post('/api/operator/stop',{job_id:job});await refreshOperator()}catch(e){alert(e.message)}};

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
