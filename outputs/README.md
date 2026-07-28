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

Run counts and generated timestamps are intentionally omitted here because this
tracked file is a directory contract, not a snapshot of local ignored outputs.
Curated evidence belongs under `data/sample_outputs/`.
