# Direct gas credit TRAIN outcome

The registered Linux CPU snapshot run completed 4,096 TRAIN decisions and four PPO updates with `direct_gas_reward=8.0`. It used the same initial actor, seed, TRAIN split, action range, optimizer rates and zero teacher warmup as the earlier high-head-rate zero-credit control. The snapshot avoided concurrent changes to unrelated shared source files; the plan records its exact executable and input hashes.

| TRAIN update | Mean gas | Mean speed | Completed episodes | Finishes | Off-track endings |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.06782 | 34.72 | 3 | 0 | 3 |
| 2 | 0.06508 | 27.87 | 2 | 1 | 1 |
| 3 | 0.06773 | 30.11 | 3 | 1 | 2 |
| 4 | 0.07098 | 24.77 | 2 | 0 | 2 |

The late mean gas of the zero-credit high-head-rate control was 0.07198. The new candidate did not meet its preregistered +0.02 action endpoint, and 8/10 observed TRAIN episode endings were off-track. The control's corresponding full episode-end distribution was not preserved in its checkpoint, so no matched completion comparison is claimed. The candidate is rejected as a mechanism activation probe. TUNE and all protected cells remain unopened. The previous selected ZIP and official site state are unchanged.

The v2 `report` command rejected the post-run gate transition because the ephemeral snapshot's full executable identity could not be reconstructed after another shared-source edit. The proposed three-gate result is in `gates-snapshot.json`, but this run has **no CLI-recorded REJECT transition or integration report**. Its raw run manifest and events remain intact. Future isolated runs must retain their exact source snapshot until gate reporting is complete.
