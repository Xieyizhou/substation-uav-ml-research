(function (root) {
  'use strict';

  function signature(scenarios) {
    return JSON.stringify(scenarios.map(row => [row.scenario_id, row.dataset_role]));
  }

  function sync(select, scenarios) {
    const nextSignature = signature(scenarios);
    if (select.dataset.scenarioSignature === nextSignature) return false;

    const previousValue = select.value;
    const options = scenarios.map(row => {
      const option = document.createElement('option');
      option.value = row.scenario_id;
      option.textContent = `${row.scenario_id} · ${row.dataset_role}`;
      return option;
    });
    select.replaceChildren(...options);
    select.dataset.scenarioSignature = nextSignature;

    if (scenarios.some(row => row.scenario_id === previousValue)) {
      select.value = previousValue;
    } else if (scenarios.length) {
      select.value = scenarios[0].scenario_id;
    }
    return true;
  }

  root.SandboxScenarioSelector = {signature, sync};
})(globalThis);
