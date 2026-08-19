(function(root){
  const routes=new Set([
    'fly/run','fly/recordings','model/datasets','model/train',
    'model/runs','model/inference','results/visual','results/lidar',
    'results/acceptance','activity/jobs','activity/logs',
    'activity/environment','activity/storage',
  ]);
  const defaults={home:'model/datasets',fly:'fly/run',model:'model/datasets',results:'results/visual',activity:'activity/jobs'};
  const legacy={
    setup:'model/datasets',overview:'model/datasets',experiments:'model/train',research:'results/visual',
    lidar:'results/lidar',preflight:'results/lidar',frames:'fly/recordings',
    operator:'activity/jobs',logs:'activity/logs',doctor:'activity/environment',
  };
  const guidance={
    simulator_start_failed:'Check the simulator log and runtime compatibility before retrying.',
    flight_timeout:'Inspect the flight log and confirm the route reached a terminal state.',
    output_budget_exceeded:'Free managed output space, then start a new job.',
    workbench_failed:'Open the selected job log; a checkpoint may be resumable from Model Lab.',
  };
  function normalize(hash,fallback='model/datasets'){
    const value=decodeURIComponent(String(hash||'').replace(/^#/,''));
    if(routes.has(value))return value;
    if(legacy[value])return legacy[value];
    if(defaults[value])return defaults[value];
    return routes.has(fallback)?fallback:'model/datasets';
  }
  root.SandboxUIState={
    routeFromHash:normalize,
    primaryFor(route){return normalize(`#${route}`).split('/')[0]},
    failureGuidance(code){return guidance[code]||'Open the selected job log and inspect its diagnostics before retrying.'},
  };
})(typeof globalThis==='undefined'?this:globalThis);
