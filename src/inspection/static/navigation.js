(function(root){
  function activate(route,persist=true){
    const canonical=SandboxUIState.routeFromHash(`#${route}`),primary=SandboxUIState.primaryFor(canonical);
    const panel=document.getElementById(primary),top=document.querySelector(`.primary-nav [data-route^="${primary}"]`);
    if(!panel||!top)return false;
    document.querySelectorAll('.primary-nav .tab').forEach(node=>{const selected=node===top;node.classList.toggle('active',selected);node.setAttribute('aria-selected',String(selected));node.tabIndex=selected?0:-1});
    document.querySelectorAll('main > .panel').forEach(node=>node.classList.toggle('active',node===panel));
    document.querySelectorAll('.secondary-nav [data-route]').forEach(node=>{const selected=node.dataset.route===canonical;node.classList.toggle('active',selected);node.setAttribute('aria-selected',String(selected));node.tabIndex=selected?0:-1});
    document.querySelectorAll('.subview').forEach(node=>node.classList.toggle('active',node.dataset.view===canonical));
    if(location.hash!==`#${canonical}`){
      const method=persist?'pushState':'replaceState';history[method](null,'',`#${canonical}`);
    }
    document.title=`${top.textContent.trim()} · UAV Research Sandbox`;
    return true;
  }
  document.querySelectorAll('[data-route]').forEach(button=>button.addEventListener('click',()=>activate(button.dataset.route)));
  window.addEventListener('hashchange',()=>activate(SandboxUIState.routeFromHash(location.hash),false));
  root.activateRoute=activate;
  activate(SandboxUIState.routeFromHash(location.hash),false);
})(typeof globalThis==='undefined'?this:globalThis);
