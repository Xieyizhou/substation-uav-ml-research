# Different magnetic tables produced a ground heading offset

Historical ground comparison: 2026-10-22. [Selected measurements and source identities](../../data/sample_outputs/case_studies/shared-magnetic-model.json).

A planned backup-IMU-loss experiment hit the original one-degree attitude guard **before fault injection**. No IMU was removed. Investigation found that the installed Gazebo and bound PX4 models predicted different magnetic directions at the same map location, latitude **47.3979710577°**, longitude **8.5461637398°**.

| Declination at the map location | Degrees |
| --- | ---: |
| Gazebo 8.14.0 table | 2.563849 |
| Bound PX4 table | 3.405765 |
| Difference | 0.841916 |

The original failed run's ground window contained **3,750** exact ULog/ECM attitude pairs, with mean heading error **0.853167°**. That is consistent with a systematic model offset. It does not establish that the magnetic mismatch explains every later dynamic threshold crossing.

The dated experiment built a separate, explicitly selected simulator magnetometer candidate using the PX4 magnetic table. It retained the coordinate and sensor-unit contract, noise and update rate, rather than adjusting observations from the measured attitude error. Gazebo's [same-version magnetometer implementation](https://github.com/gazebosim/gz-sim/blob/gz-sim8_8.14.0/src/systems/magnetometer/Magnetometer.cc) queries its internal tables during updates, so changing only the world's fixed magnetic-field vector was not sufficient for this setup.

| Full unarmed SDK chain | Original table | Shared-table candidate |
| --- | ---: | ---: |
| Ground observations | 75 | 75 |
| Mean heading error | 0.797931° | −0.000344° |
| Maximum orientation error | 0.814663° | 0.030275° |
| Protected parameters checked | 517 | 517 |

Both comparisons passed the original 20 ms pairing, 150 ms age and one-degree ground guards. Both refused ARM, contained no ARM/TAKEOFF commands and retained `failed` mission status. The measured mean-heading reduction was **0.798275°**. The smaller 75-sample window is different from the earlier 3,750-pair window; their means must not be forced to agree.

This supports checking physical-model consistency before diagnosing estimator transport timing. It is a ground result. The original installed magnetometer library was not explicitly SHA-bound by the earlier worker, so complete historical loading proof cannot be reconstructed from a later library snapshot. Larger yaw resets and the full cause of later airborne crossings remained separate questions. Subsequent dated experiments require their own evidence; this excerpt neither grants flight qualification nor integrates the October candidate into this source snapshot.
