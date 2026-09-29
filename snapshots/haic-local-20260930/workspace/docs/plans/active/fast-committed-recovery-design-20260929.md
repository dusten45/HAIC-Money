# Preserve acceleration gains by resolving avoidance and impact state

Goal unchanged: better completion and ultimately half lap time. Three high-acceleration architecture cycles did not produce eligible completion improvements; preserve their checkpoints and pivot to coordinated fast acceleration plus avoidance/recovery state. Do not lower the speed window to fix failure. Partial advancement requires6/6 against original control and lower median matched completed time; full user target still every matched ratio<=.5 and broader completion evidence.

Evidence: geometric-passing launch_guarded1:38300 finishes21.24vs22.10;1:38302fails66.9%. Before first collision, steering changes+.054,-.355,-.622,+.172,+.313 at speed44-54. Damage then rises eachdecision. Recent-brake veto may suppress collision recovery: current code tests anyoflast4brakes, while speed measurement is averaged over2frames. Need separate candidate interventions to avoid attributing success to multiple repairs.

Common acceleration: keep exactly previous launch_guarded schedule (gas1untilHUD40, gasfloor.55 on full visible straight withspread<3,nearerror<2,noobstacle,speed<70; otherwisebaseline). Controls: unchangedpreview_row_repair and exactpriorlaunch_guarded fast_control. Four candidates:
1 commit_side: retain initial camera-relative passing side until two absent frames; replace only obstacle steering term, same road response.
2 passing_bearing: retain side and use metric passing point inside observed road bounds, converting displacement to steering bearing instead of fixedkick. No guarantee when insufficient width; mark fallback.
3 shock_recovery: use single-frame HUD speed drop minus conservative braking allowance240*lastissuedbrake*.08. If residual>10, retain12decision recovery and remove obstacle steering while preserving road steering. No privilegedcollisioninput. This explicitly tests the recent-brake masking hypothesis, and may have false positives.
4 continuous_avoidance: bound change in obstacle steering component to.08/decision, retain baseline road steering. This is actuator continuity rather than side commitment.

Geometry missing: reconstruct row42 only as existingrowrepair if54present andatleast2rows; otherwise don't fabricate road response. Snapshot actual current inherited road steering before changing obstacle terms. Track requested obstacle term, appliedterm, commitment,shock,memory,actual actiondifference,brakehistory. Policy pixels only. Evaluator telemetry separate.

Six arms x six consumedTRAINcells=36episodes,1200decisions,2CPU2GiB1800s. Same resetconditions, launchprefixintentionaldifference. Compare candidate to bothcontrols but use originalcontrolcompletion6/6as eligibility. No coefficient sweep, no freshvalidationclaim, no externalactions. Review first; register exact manifest and separate standing design/execution records. Finish andintegrate previous frozenrun before code edits.
