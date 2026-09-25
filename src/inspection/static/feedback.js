/* Explicit image review; unreviewed predictions never register as training labels. */
(() => {
  const el = name => document.querySelector('#feedback-' + name);
  let collections = [], collection = null, detail = null, sampleId = '', boxes = [], selected = -1;
  let active = null, loading = false, sequence = 0, drawing = null, collectRun = null, expectedReviewIdentity = null;
  const drafts = new Map();
  const names = {transformer:'变压器', switchgear:'开关柜', capacitor_bank:'电容器组', reactor:'电抗器'};
  const statusName = {unreviewed:'待审核', accepted:'已接纳', rejected:'已拒绝'};
  const copy = value => JSON.parse(JSON.stringify(value));
  function message(text, error=false) { el('status').textContent=text; el('status').classList.toggle('feedback-error',error); }
  function key() { return collection ? collection.collection_id + '/' + sampleId : ''; }
  function remember() {
    if (detail) drafts.set(key(), {boxes:copy(boxes), noTarget:el('no-target').checked, complete:el('complete').checked,
      reviewer:el('reviewer').value, note:el('note').value, expectedIdentity:expectedReviewIdentity});
  }
  function controls() {
    const busy = Boolean(active) || loading || !operatorToken;
    el('collect').disabled = busy || profileState?.profile_id !== 'development';
    for (const name of ['accept','reject','draft','add','clear']) el(name).disabled = loading || !detail || !operatorToken;
    el('delete').disabled = loading || selected < 0;
    const previous = el('base').value;
    if (workbenchState) {
      const eligible = workbenchState.datasets.filter(d=>Object.keys(names).every(c=>(d.class_counts?.validation?.[c]||0)>0));
      const signature = eligible.map(d=>d.dataset_id).join('|');
      if (el('base').dataset.signature !== signature) {
        el('base').innerHTML=eligible.map(d=>`<option value="${esc(d.dataset_id)}">${esc(d.dataset_id)}</option>`).join('');
        el('base').dataset.signature=signature;
        if (eligible.some(d=>d.dataset_id===previous)) el('base').value=previous;
      }
    }
    el('register').disabled = busy || !collection || !el('base').value || !collection.samples.some(s=>s.status==='accepted') || collection.samples.some(s=>s.status==='unreviewed');
  }
  function renderList() {
    el('samples').innerHTML=collection.samples.map((s,i)=>`<option value="${s.sample_id}">${i+1}. ${statusName[s.status]}${s.boxes==null?'':` · ${s.boxes} 框`}</option>`).join('');
    el('samples').value=sampleId;
    const remaining=collection.samples.filter(s=>s.status==='unreviewed').length;
    el('counts').textContent=`${collection.samples.length} 张 · ${remaining} 待审核 · ${collection.samples.filter(s=>s.status==='accepted').length} 已接纳`;
  }
  function predictions() {
    return (detail?.sample.predictions || []).filter(p=>names[p.class_name] && (p.xyxy||p.bbox)?.length===4)
      .map(p=>({class_name:p.class_name,xyxy:[...(p.xyxy||p.bbox)]}))
      .filter(p=>p.xyxy.every(Number.isFinite));
  }
  function renderCanvas() {
    if (!detail) return;
    const f=detail.sample.frame, font=Math.max(10,f.width/50);
    const rectangle=(b,index,guess=false)=>{
      const [x1,y1,x2,y2]=b.xyxy;
      return `<g ${guess?'':`data-box="${index}"`}><rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" fill="none" stroke="${guess?'#e5a52e':index===selected?'#ff5858':'#00ddac'}" stroke-width="2" vector-effect="non-scaling-stroke" ${guess?'stroke-dasharray="5 5"':''}/><text x="${x1+2}" y="${Math.max(font,y1-3)}" font-size="${font}" fill="${guess?'#ffd273':'#00ddac'}" stroke="#12231c" stroke-width=".4">${esc(names[b.class_name])}${guess?' · 预测':''}</text></g>`;
    };
    el('canvas').setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);
    el('canvas').style.aspectRatio=`${f.width}/${f.height}`;
    const href=`/api/feedback/${collection.collection_id}/${sampleId}/image`;
    if(el('canvas').querySelector('image')?.getAttribute('href')!==href)
      el('canvas').innerHTML=`<image href="${href}" width="${f.width}" height="${f.height}"/><g data-overlays="true"></g>`;
    el('canvas').querySelector('[data-overlays]').innerHTML=(el('show-predictions').checked?predictions().map((b,i)=>rectangle(b,i,true)).join(''):'')+
      boxes.map((b,i)=>rectangle(b,i)).join('')+(drawing?rectangle({class_name:el('class').value,xyxy:drawing.box},-1):'');
    el('boxes').innerHTML=boxes.map((b,i)=>`<button type="button" data-index="${i}" aria-pressed="${i===selected}">${i+1}. ${esc(names[b.class_name])}</button>`).join('');
    el('boxes').querySelectorAll('button').forEach(button=>button.onclick=()=>chooseBox(Number(button.dataset.index)));
    el('delete').disabled=selected<0;el('add').textContent=selected<0?'添加框':'更新选中框';
  }
  function chooseBox(index) {
    selected=index;
    if (index>=0) { el('class').value=boxes[index].class_name; ['x1','y1','x2','y2'].forEach((n,i)=>el(n).value=boxes[index].xyxy[i]); }
    renderCanvas();
  }
  function changed() { el('complete').checked=false; el('no-target').checked=boxes.length===0 && el('no-target').checked; remember(); renderCanvas(); }
  async function loadSample(id) {
    remember(); const ticket=++sequence; sampleId=id; detail=null; selected=-1; loading=true; controls();
    try {
      const value=await api(`/api/feedback/${collection.collection_id}/${id}`);
      if(ticket!==sequence)return;
      detail=value; const draft=drafts.get(key()), review=value.review;
      expectedReviewIdentity=draft?draft.expectedIdentity:review?.review_identity_sha256||null;
      boxes=copy(draft?.boxes||review?.boxes||[]);
      el('no-target').checked=draft?.noTarget??review?.no_target??false;
      el('complete').checked=draft?.complete??review?.complete_annotation??false;
      el('reviewer').value=draft?.reviewer||review?.reviewer||el('reviewer').value||'本机操作员';
      el('note').value=draft?.note??review?.note??'';
      renderList(); renderCanvas(); message(review?`已有审核：${statusName[review.decision]}；编辑后需重新提交。`:'预测仅作为提示。请标注完整画面，或明确拒绝此样本。');
      if(draft && draft.expectedIdentity!==(review?.review_identity_sha256||null))message('此草稿对应的审核已在其他窗口改变。请放弃草稿并重载，再提交修改。',true);
    } catch(error) { if(ticket===sequence)message(error.message,true); }
    finally { if(ticket===sequence){loading=false;controls();} }
  }
  async function loadCollection(id) {
    remember(); const ticket=++sequence; detail=null; loading=true; controls();
    try { const value=await api('/api/feedback/'+id);if(ticket!==sequence)return;collection=value; sampleId=''; renderList();
      const first=collection.samples.find(s=>s.status==='unreviewed')||collection.samples[0];
      if(first)await loadSample(first.sample_id);else message('采集没有可审核的图片。');
    } catch(error) {if(ticket===sequence){collection=null;message(error.message,true);}}
    finally {if(ticket===sequence){loading=false;controls();}}
  }
  async function refresh() {
    try {
      const value=await api('/api/feedback'); collections=value.collections.filter(c=>c.state==='complete');
      const old=el('collection').value;
      el('collection').innerHTML=collections.map(c=>`<option value="${c.collection_id}">${esc(c.name)} · ${c.sample_count} 张</option>`).join('');
      if(collections.some(c=>c.collection_id===old))el('collection').value=old;
      const captured=collectRun && collections.find(c=>c.name===collectRun);
      if(captured){collectRun=null;el('collection').value=captured.collection_id;await loadCollection(captured.collection_id);}
      else if(!collection && !loading && collections.length)await loadCollection(el('collection').value);
      controls();
    } catch(error){message(error.message,true);}
  }
  async function save(accepted) {
    if(!detail)return;
    const review={decision:accepted?'accepted':'rejected',boxes:copy(boxes),no_target:el('no-target').checked,
      complete_annotation:el('complete').checked,reviewer:el('reviewer').value.trim(),note:el('note').value.trim()};
    loading=true;controls();
    try {
      const result=await post('/api/feedback/review',{collection_id:collection.collection_id,sample_id:sampleId,review,expected_identity:expectedReviewIdentity});
      detail.review=result;expectedReviewIdentity=result.review_identity_sha256;drafts.delete(key());
      const row=collection.samples.find(s=>s.sample_id===sampleId);row.status=result.decision;row.boxes=result.boxes.length;
      renderList();message(`已保存：${statusName[result.decision]}。原始采集与预测保持可追溯。`);
    }catch(error){message(error.message,true);}
    finally{loading=false;controls();}
  }
  el('accept').onclick=()=>save(true);el('reject').onclick=()=>save(false);
  el('collection').onchange=()=>loadCollection(el('collection').value);
  el('samples').onchange=()=>loadSample(el('samples').value);
  el('refresh').onclick=refresh;
  el('reload').onclick=()=>{if(!detail)return;drafts.delete(key());detail=null;loadSample(sampleId);};
  el('next').onclick=()=>{const rows=collection?.samples||[],i=rows.findIndex(s=>s.sample_id===sampleId);if(rows[i+1])loadSample(rows[i+1].sample_id);};
  el('draft').onclick=()=>{boxes=predictions();selected=-1;el('no-target').checked=false;changed();message('预测已复制为草稿，仍需纠正、补全并提交审核。');};
  el('show-predictions').onchange=renderCanvas;
  el('clear').onclick=()=>{boxes=[];selected=-1;changed();};
  el('delete').onclick=()=>{if(selected>=0){boxes.splice(selected,1);selected=-1;changed();}};
  el('add').onclick=()=>{const box={class_name:el('class').value,xyxy:['x1','y1','x2','y2'].map(n=>Number(el(n).value))};
    const [a,b,c,d]=box.xyxy,f=detail.sample.frame;if(!box.xyxy.every(Number.isFinite)||!(0<=a&&a<c&&c<=f.width&&0<=b&&b<d&&d<=f.height)){message('框必须在图片内，且有正的宽度和高度。',true);return;}
    if(selected<0)boxes.push(box);else boxes[selected]=box;el('no-target').checked=false;changed();};
  el('no-target').onchange=()=>{if(el('no-target').checked){boxes=[];selected=-1;}changed();};
  el('base').onchange=controls;
  function point(event){const r=el('canvas').getBoundingClientRect(),f=detail.sample.frame;return [Math.max(0,Math.min(f.width,(event.clientX-r.left)*f.width/r.width)),Math.max(0,Math.min(f.height,(event.clientY-r.top)*f.height/r.height))].map(v=>Math.round(v*10)/10);}
  el('canvas').onpointerdown=event=>{if(!detail||loading||event.button!==0)return;const p=point(event);drawing={start:p,box:[...p,...p]};el('canvas').setPointerCapture(event.pointerId);};
  el('canvas').onpointermove=event=>{if(!drawing)return;const p=point(event),a=drawing.start;drawing.box=[Math.min(a[0],p[0]),Math.min(a[1],p[1]),Math.max(a[0],p[0]),Math.max(a[1],p[1])];renderCanvas();};
  el('canvas').onpointerup=()=>{if(!drawing)return;const b=drawing.box;drawing=null;if(b[2]-b[0]>=2&&b[3]-b[1]>=2){boxes.push({class_name:el('class').value,xyxy:b});selected=boxes.length-1;el('no-target').checked=false;changed();chooseBox(selected);}else renderCanvas();};
  el('canvas').onpointercancel=()=>{drawing=null;renderCanvas();};
  el('collect').onclick=()=>{
    location.hash='fly/run';
    message('请在 Flight Console 选择模型并执行任务。飞行反馈会自动保存，结束后返回这里刷新并审核。');
  };
  el('register').onclick=async()=>{
    const parameters={dataset_id:el('dataset-id').value.trim(),base_dataset_id:el('base').value,collection_ids:[collection.collection_id]};
    if(!await requestConfirmation({title:'注册已审核数据集？',detail:'新增反馈整组加入训练；从基础数据集固定抽取最多 256 张训练回放和 64 张验证图。拒绝样本不进入训练。',facts:{数据集:parameters.dataset_id,基础数据集:parameters.base_dataset_id,接纳图片:collection.samples.filter(s=>s.status==='accepted').length},confirmLabel:'注册数据集'}))return;
    try{await post('/api/operator/start',{action:'workbench-feedback-register',parameters});message('正在注册。完成后可以在 Training Studio 选择这个数据集。');await refreshOperator();}
    catch(error){message(error.message,true);}
  };
  globalThis.renderFeedbackOperator=job=>{const ended=active&&!job;active=job;controls();if(ended)refresh();};
  refresh();setInterval(refresh,10000);
})();
