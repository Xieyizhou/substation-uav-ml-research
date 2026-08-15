let setupFirstObservation=true;
function renderSetup(value){
  const complete=value.required_count?Math.round(100*value.required_ready_count/value.required_count):100;
  $('#setup-progress').innerHTML=`<strong>${value.required_ready_count} / ${value.required_count}</strong><span>required items ready</span><div class="progress-track"><div class="progress-fill" style="width:${complete}%"></div></div>`;
  $('#setup-items').innerHTML=value.items.map(item=>{const needsAction=item.status!=='ready';return `<article class="setup-item ${item.status}"><div class="setup-item-head"><span class="badge ${item.status==='ready'?'pass':item.required?'failure':'warning'}">${esc(item.status)}</span><span>${item.required?'Required':'Optional'}</span></div><h3>${esc(item.title)}</h3><p>${esc(item.detail)}</p>${needsAction&&item.action?`<small>${esc(item.action)}</small>`:''}<div class="setup-actions">${needsAction&&item.command?`<button class="copy-command" data-command="${esc(item.command)}">Copy command</button>`:''}${item.docs_url?`<a href="${esc(item.docs_url)}" target="_blank" rel="noreferrer">Official guide</a>`:''}</div></article>`}).join('');
  $('#setup-next-step').textContent=value.next_step;
  document.querySelectorAll('.copy-command').forEach(button=>button.onclick=async()=>{try{await navigator.clipboard.writeText(button.dataset.command);button.textContent='Copied'}catch{actionError('Clipboard access is unavailable. Copy the command from Sandbox doctor.')}});
  if(setupFirstObservation&&!value.ready){activateTab('setup')}setupFirstObservation=false;
}
window.addEventListener('DOMContentLoaded',()=>{$('#setup-check').onclick=()=>refresh().catch(error=>actionError(error.message))});
