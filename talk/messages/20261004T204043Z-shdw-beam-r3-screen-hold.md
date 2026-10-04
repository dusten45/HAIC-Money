# Beam R3 improves required laps but fails one old regression
- Message ID: 20261004T204043Z-shdw-beam-r3-screen-hold
- Type: result
- Author/session: shdw
- Scope: agents/apex_2026/v2/beam_agent.py

Frozen source aef90410f8ad7f3f022a7c5a07c9ba87b997304964a63e5ca0601d8e60e7e8fa completed the prespecified consumed six-cell screen. Required four finish damage-free at 14.04/17.24/15.80/14.94s (mean15.505s versus frozen P1 18.53s). Old failed track2/4031370700 finishes13.48s damage0; old failed track3/4111953688 retires off_track at progress.616099, damage.4. Five of six finish and qualify, all six satisfy measured hardresource limits; maxact4.037592s. Original10–13s goal0/6. Full consumed24 gate is CLOSED. Runtime HOLD, no adoption or freshholdout claim.

Primary receipts: agents/apex_2026/v2/results/beam-r3/; ledger beam-r3-screen.json. Previous r1 six executionfailures and r2 six drivingfailures remain preserved:18 screenresets over six unique consumedcells, plus one deterministic diagnostic replay of160 recordedactions. Replay positionparity checked1e-4 for all160, and beforeobservations/truth120–160 saved; full249/fullstate parity is not claimed.

Firstcollision occurs138/139, preceding stop149 and coastprocrastination154–249. Observableinitialization80msposition errors132–138 are.065–.148m. Aftercollision140–144 invalidflow carries undersized slip, residual grows.42–1.05m. Footprint/mask firstcollision diagnosis remains independent work. Runtime uses approximate25point footprint, known to miss interior occupiedpixels and maximally steered wheel extent; RPM ambiguousflag is diagnostically overinclusive. These findings do not justify silently fixing the frozen batch or counting it as six successes.
