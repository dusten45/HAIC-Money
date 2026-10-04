# Camera road graph diagnostic and rejected component candidate

All analysis uses fresh saved prefix frames from the corridor V1 benchmark.
`graph_geometry.py` reconstructs the official forward track only for offline
attribution using pre-action trace pose (frame i uses trace i-1). No simulator
state is supplied to camera inference. It resets maps to recover geometry and
initial pose but performs no driving episodes.

Track 2 has no mean parsed-center distance over 4 m until step 174. At that
step the camera car is already outside the lower visible road; the row scanner
then crosses a grass gap into a disconnected upper road. This is a recovery
problem, not demonstrated cause of the initial road exit. Later large errors
include reverse-facing motion and must not be treated as clean pre-exit evidence.

Track 4 tight bends at steps 64–66 and 98–100 turn nearly horizontal before
exiting the left image boundary. Single-valued row centers and approximate
clipped-width correction create mean center errors 2.6–2.9 m and maxima 6–7 m.
The clipped center representation is a concrete geometry risk.

`fast_graph_agent.py` embeds the exact frozen corridor V1 source and changes
only road selection to the asphalt pixel graph component connected to an ego
neighborhood. Eight-neighbor pixel adjacency prevents a disconnected road
from extending a parsed route. Eligible seeds are on contiguous asphalt runs
of at least eight pixels, excluding isolated gray car fragments. The saved
step-174 regression fails on the baseline (34.10 m false forward extent) and
passes on this candidate. Seven focused graph/corridor tests pass.

The initial seed implementation failed the saved-frame regression by selecting
an isolated car fragment. An accidentally started screen was interrupted at
the shell (exit 130), but evaluator workers completed all four source-bound
cells and the aggregate receipt. This was discovered during final artifact
inspection; it is a complete fresh benchmark, not a partial result. Its
source is preserved in ignored artifacts and a zero-context patch binds it
to the corrected source. Initial graph V1 completed 15.04 / 24.22 / 19.86 /
19.96 s, contacts 0 / 1 / 0 / 2. It fails the camera regression and strict
pace goal and is not adopted. Correcting the camera seed worsened driving,
so image-model correctness cannot be equated with policy improvement.

Fresh complete corrected graph V2 mandatory benchmark:

| track/seed | actual lap | progress | contacts |
| --- | --- | --- | --- |
| 1 / 516237 | 14.84 s | 1.0 | 0 |
| 2 / 644062 | DNF | 0.693452 | 0 |
| 3 / 1007 | DNF | 1.0 | 2 |
| 4 / 18800 | DNF | 0.530466 | 0 |

This is rejected and not adopted. Progress 1.0 on track 3 is not an official
finish. The original 10–13 s goal is unmet. No holdout is opened.

A separate `graph_ridge.py` offline prototype uses only camera pixels to
produce a parametric route from metric distance to observed non-road pixels.
Small enclosed car/obstacle holes are filled; image edges remain unknown, not
assumed grass. A traced ridge follows the visible nearly horizontal bends
without the false row-center displacement. SciPy computes the exact distance
field as a research oracle; this prototype is not an inference submission or
a validated vehicle controller. NumPy implementation and causal controlled
validation are required before drawing driving conclusions.
