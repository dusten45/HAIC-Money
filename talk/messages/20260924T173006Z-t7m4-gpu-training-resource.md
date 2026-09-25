# Concurrent GPU Research Workload

- Message ID: `20260924T173006Z-t7m4-gpu-training-resource`
- Type: INFO
- Author/session: `t7m4`
- Written: 2026-09-24T17:30:06Z
- Reply to: `20260924T172724Z-t7m4-teacher-coverage-risk`
- Evidence: measured
- Status: open

Both teacher datasets passed A3 (learner 0: 16,384 decisions, 33 complete
episodes, 6 finished geometry seeds; learner 1: 16,384 decisions, 41 complete
episodes, exactly 4 finished geometry seeds). Before A2 training, `nvidia-smi`
showed the RTX 5070 Ti with 15.9 GiB free and 35% utilization from a concurrent
Dreamer open-loop diagnostic (`PID 383093`, 396 MiB VRAM); the DrQ matrix runs
four arms sequentially and fails closed below 8 GiB free. This is shared-device
wall-clock contention only; environments, seeds, replay and RNG streams remain
independent. The study remains internal, not official HAIC performance.
