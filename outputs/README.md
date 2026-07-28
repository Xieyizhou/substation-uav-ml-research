# Outputs

New analysis outputs are stage-scoped. Do not write new `as_*` run folders directly under `outputs/`.

## Stages

| stage | summary location |
|---|---|
| `static_astar` | `outputs/01_static_astar/summaries/` |
| `perception_response` | `outputs/02_perception_response/summaries/` |
| `replan_log_only` | `outputs/03_replan_log_only/summaries/` |
| `active_replan` | `outputs/04_active_replan/summaries/` |

## Comparisons

- Cross-stage comparison outputs: `outputs/comparisons`
- Use `python main.py report compare` for intentional landmark comparisons across stages.

## ML research

- Registry: `outputs/research/registry.sqlite`
- Study queues/results: `outputs/research/study_results/`
- Randomized worlds: `outputs/research/worlds/`
- Predictions and benchmarks: `outputs/research/predictions/`

The complete `outputs/` tree is ignored. Only reviewed JSON/CSV/Markdown
summaries may be copied intentionally into `data/sample_outputs/`.

Run counts and generated timestamps are intentionally omitted here because this
tracked file is a directory contract, not a snapshot of local ignored outputs.
Curated evidence belongs under `data/sample_outputs/`.
