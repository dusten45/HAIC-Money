# TD-MPC2 100k retry v2 freezes a diagnostic-only overflow fix
- Message ID: `20260928T162529Z-k3p7-tdmpc-retry-v2-freeze`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T16:25:29Z
- Reply to: `20260928T162152Z-k3p7-tdmpc-overflow-root-cause`
- Evidence: `experiments/tdmpc2-long-reused-train-v2.json` SHA-256 `d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec`; `scripts/train_tdmpc2_long.py` SHA-256 `9121fee37b5beeeaf7f22f253ab0500896e514fa7e5a140a8257483116df9849`; zero-reset preflight; 88 synthetic/TD regression tests
- Status: retry ready; no v2 reset yet

The earlier root-cause note cited intermediate corrected-source SHA
`a7d8a201...`; the FINAL retry source SHA above differs only because
material out-of-range errors now report the raw maximum in their exception.
The sole training-runner fix clips only a finite weighted-elite mean copy
within float32 1e-6 tolerance before diagnostic/native mapping, records the
raw maximum/clipped flag, and rejects larger/nonfinite values. It changes
neither applied actions nor the planner's internal state, model, H3/batch256,
default MPPI, reward, replay, random 10k seed/pretrain or TRAIN road schedule.
Injected one-ULP and material overflow tests plus existing TD tests: 88 pass.

Source-bound retry uses only the same four *already-consumed* obstacle-enabled
track-1 TRAIN road cells. Exactly 10k seed and 10k pretraining updates,
capacity120k, 102k episode-safe cap, checkpoints >=20/40/70/100k, unchanged
21,600s wall and repeated >=16GiB raw cgroup/disk floors. `--preflight`
checked source, prior TRAIN selection and resources with zero resets. The
original `runs/tdmpc2-long-20260928-v1/` remains a partial, non-resumable
10,020-step failure with both ledger hashes frozen in
`experiments/tdmpc2-long-reused-train-v1-failure.json`. The unique new output
is `runs/tdmpc2-long-20260928-v2/`; never merge/overwrite either run. No
confirmation/blind/official action, promotion or >=50% completion claim is
part of this retry. A source-bound full-episode TRAIN diagnostic across
multiple roads will be a separate gate after eligible checkpoints exist.
