(function(root){
  const tabs=new Set(['setup','overview','research','lidar','experiments','preflight','doctor','operator','logs','frames']);
  const guidance={
    simulator_start_failed:'Check the simulator log and runtime compatibility before retrying.',
    flight_timeout:'Inspect the flight log and confirm the route reached a terminal state.',
    output_budget_exceeded:'Free managed output space, then start a new job.',
    workbench_failed:'Open the selected job log; a checkpoint may be resumable from Workbench.',
  };
  root.SandboxUIState={
    tabFromHash(hash,fallback='overview'){const value=String(hash||'').replace(/^#/,'');return tabs.has(value)?value:fallback},
    failureGuidance(code){return guidance[code]||'Open the selected job log and inspect its diagnostics before retrying.'},
  };
})(typeof globalThis==='undefined'?this:globalThis);
