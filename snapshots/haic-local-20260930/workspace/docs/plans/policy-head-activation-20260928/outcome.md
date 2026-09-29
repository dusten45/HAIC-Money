# Policy-head learning activation outcome

Two TRAIN-only arms completed in Linux Python 3.11 CPU, 4,096 decisions and four PPO updates each. Both initialized actor weights from the same immutable v2 checkpoint and started a fresh optimizer. The only planned learning difference was policy-mean LR: control `2e-6`, candidate `2e-4`. Teacher warmup was explicitly zero. No TUNE or protected cell was opened.

| Late TRAIN rollout measure | Control | Candidate |
|---|---:|---:|
| Mean gas | 0.07787 | 0.07198 |
| Mean speed | 24.99 | 32.14 |
| Maximum gas | 0.22719 | 0.22985 |
| Policy-mean weight relative L2 change from shared start | 0.073% | 4.302% |

The higher head rate moved the policy weights, but it did **not** produce the registered `+0.02` mean-gas change. Mechanism activation is `FAIL`. The speed numbers come from different TRAIN trajectories, so they do not establish a speed or lap-time improvement. The candidate does not advance to TUNE. This suggests the bottleneck is the direction or quality of the action-learning signal, beyond simply how fast the action head updates; it does not prove a unique root cause.

The first control run was stopped after discovering that actor-only initialization silently enabled the default three teacher warmup epochs. That execution is infrastructure-invalid and is not a performance cycle. An earlier r2 control plan was blocked before execution when another task changed shared source bytes. The successful matched arms are candidate `policy-head-activation-candidate-r2-20260928` and control `policy-head-activation-control-r3-20260928`, with their own immutable run manifests and checkpoint files under the v2 roots. The previously selected ZIP and official site state were not changed.
