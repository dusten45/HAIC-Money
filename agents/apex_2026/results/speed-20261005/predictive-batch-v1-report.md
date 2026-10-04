# Vectorized four-tire model

New NumPy-only helper `predictive_batch_model.py` is frozen at
`8ae7239b81d9b41e649e2ef704c3a6bbf3a8b2a933ed499b7bf13f566cb82d53`.
The scalar oracle remains unchanged at
`84a91143f8d42db7cf1007586c5c13038c6e0e6d371624f262abcaba26e46064`.
No existing planner, observer, standalone agent or official physics was edited.

`pack_states(states)` creates independent row-aligned float64 arrays;
`predict_batch(batch, physical_commands)` advances one .02 second raw tick;
`unpack_state(batch, index)` restores a copied scalar state. Commands are
physical joint targets, gas and brake. Caller converts legal steering sign.
Finite raw commands are not clamped, preserving the scalar model's branches.
Normalized float32 values are promoted to float64 before comparison, so
float32 .9 remains partial while exact float32 1.0 locks before tire response.

All force and update order is preserved: immediate rear gas reduction/.1 rise,
engine omega before braking, old-joint tire velocity/force frames, 100 m/s
individual wheel caps, 400 N per-tire friction, omega tire response, compound
force/torque, hull speed cap, yaw and angle update, then steering motor and
four-joint limits. Output arrays do not alias input arrays. Supported scope is
`wheel_velocity_override=None` and the existing uniform-asphalt model. Grass,
damage, contacts and privileged velocity overrides are not added.

Eight focused tests passed, including heterogeneous branches, 16/32-step scalar
recurrence, input preservation, rotated asymmetric mechanics, caps, row
permutation and float32 lock behavior. Independent review AST-isolated the
frozen scalar functions and tested ten different fixed cases for every tick
through 32: maximum state discrepancy 1.42e-14 and diagnostic discrepancy
5.97e-13, within atol 1e-10/rtol 1e-12. No formula discrepancy was found.

| Fixed workload, 216 rows | Scalar ms | Batch ms | Speedup |
| --- | ---: | ---: | ---: |
| 16 constant-control ticks | 572.56 | 10.41 | 54.98× |
| 4 control + 32 held-joint full-brake ticks | 1292.71 | 20.61 | 62.74× |

Initial packing took 1.44–1.73 ms. Every state field and diagnostic matched
exactly on these profile workloads. The profile binds all source hashes and
checks that they remained unchanged. It performs zero world steps and opens
no holdout. Speedups measure the model kernel on this host; camera geometry,
observer, policy selection and official worst-case latency are excluded.
This study establishes no completed-lap or pace-goal result.
