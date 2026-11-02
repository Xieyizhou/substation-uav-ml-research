# A port diagnostic delayed shutdown notification

Historical comparison: 2026-10-22. [Selected measurements and source identities](../../data/sample_outputs/case_studies/bounded-shutdown-diagnostics.json).

A low-altitude SITL route completed and landed, but its logger queue overflowed during cleanup. Looking only at the overall delay could have implicated the SDK shutdown. The boundary timestamps showed that the SDK child was reaped quickly; a subsequent port diagnostic consumed most of the time before the simulator was told to stop.

| Measurement | Earlier route | Bounded diagnostic route |
| --- | ---: | ---: |
| SDK stop request to reap | 1.633 ms | 1.374 ms |
| Total port diagnostic | 62,615.732 ms | 36.767 ms |
| Mission completion to simulator stop request | 63.692 s | 1.168 s |
| Queue produced / accepted | 70,287 / 57,036 | 54,450 / 54,450 |
| Queue forwarded / pending | 52,940 / 4,096 | 52,727 / 1,723 |
| Dropped / fault | 13,251 / true | 0 / false |

The earlier `lsof` command alone took **62,604.553 ms**. The historical fix reused the project's Darwin `posix_spawn` and CLOEXEC process path for port diagnostics. Three commands shared a **two-second total diagnostic budget**, recorded their owned PIDs and reap results, and stopped launching more work after the budget was exhausted. An exhausted or failed check was recorded as unknown; it could not prove a port was free.

All three diagnostic children in the later excerpt were reaped, and only owned children could be signalled. Queue capacity stayed **4,096**, and each native simulator child's original **six-second** shutdown grace stayed unchanged. The comparison improved the measured cleanup boundary without enlarging the queue or suppressing actual native publication.

Two accounting identities remain useful when reviewing the excerpt: `produced = accepted + dropped` and `accepted = forwarded + pending`. Pending entries were not retroactively written into the old ULog or counted as already persisted. The earlier route remains completed **with a queue failure**; route completion did not make the diagnostic healthy.

The old internal spawn-blocker stack was not captured, so this evidence does not prove a unique low-level cause inside `lsof` or Python process creation. It supports bounding auxiliary diagnostics and measuring each lifecycle boundary separately. Complete lifecycle logging, ordinary ARM and current-source flight qualification remain outside the excerpt.
