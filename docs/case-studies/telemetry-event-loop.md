# A provenance check blocked telemetry progress

Historical experiment: 2026-10-24. [Selected measurements and source identities](../../data/sample_outputs/case_studies/telemetry-event-loop.json).

The first run refused ARM because local-position telemetry was stale: **17.9 seconds against the original 2-second limit**. The task stayed `failed`; the actual ULog contained no ARM or TAKEOFF. Both native children exited within the existing six-second shutdown budget. This separated the pre-ARM refusal from a native cleanup failure.

| Measurement | Synchronous validation | Background validation |
| --- | ---: | ---: |
| Logger rows in the first 30 seconds | 18 | 150 |
| Maximum logger gap in that window | 17.966 s | 0.206 s |

An offline profile of the unchanged bound protocol took **17.227 seconds** to validate. In the failing run, ground-check start to refusal was **17.941 seconds**. The synchronous `validate_bound_request` call occupied the same event loop that needed to advance telemetry. The matching pause duration and the subsequent concurrency comparison support this cause for this refusal.

The dated research change awaited `asyncio.to_thread(validate_bound_request, ...)` in the admission path. It retained the full provenance validation and ran the original ground-stability, source-freshness and real-time admission gates after validation. It did not widen the two-second freshness limit. During the comparison, the ground-to-dual-trace interval was still **19.580 seconds**, but logging continued throughout it and the original ground-stability check passed.

The useful pattern is to measure event-loop progress around expensive integrity work, then preserve the validation and admission order when moving that work to a worker thread. A callback that looks like metadata validation can still contain enough file hashing to disrupt live data consumers.

Logger cadence alone does not prove every MAVSDK stream is fresh. The full historical healthy-flight audit is outside this compact excerpt, and the excerpt does not qualify this publication source for ordinary ARM. The October admission implementation and its native runtime assets have not been imported as an independently qualified flight system here. This finding also does not explain every earlier connection loss.
