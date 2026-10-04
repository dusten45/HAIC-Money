# Arc guard V1: camera regression passes, lap candidate rejected

This bounded experiment embeds the exact early-circle V1 definitions under a
class alias, then checks the base controller's actual slew-limited emitted
turn. It samples rigid vehicle corners (longitudinal -2.4/+2.6 m, lateral
±1.6 m) along a constant-turn arc over 0–8 m. If minimum camera road depth is
below 1.9 m, it tries one 8 m preview and selects it only when depth improves
and reaches 1.9 m. Selected target speed includes the actual emitted-steer
lateral-force bound. Clear-ridge selection and confidence recovery remain.
Camera corner samples do not certify tyre sweep, lag or dynamic feasibility.

Actual track-3 cameras 77 and 78 failed the initial regression on the parent
(corner depths 0.643/0 m). The new regression passes; arc-guard, early-circle
and clear-supported focused tests pass together: 14 passed. Supported clear
camera actions, late unsupported edge actions, invalid-input recovery and
reset retain their reference behavior. The zero-context source patch reverses
to the exact parent SHA, and the evaluated source stayed unchanged.

One fresh mandatory benchmark used defaults and 700 actions per cell:

| Track | Seed | Official lap | Contacts |
| --- | --- | --- | --- |
| 1 | 516237 | 15.44 s | 0 |
| 2 | 644062 | 19.50 s | 0 |
| 3 | 1007 | 19.76 s | 2 |
| 4 | 18800 | 19.78 s | 1 |

The original 13-second and early 10-second goals remain unmet. This candidate
is not adopted. The preserved earlier early-circle receipt reports
15.20/22.72/18.92/19.22 s with contacts 0/2/0/0; those results are historical
reference data, not newly repeated controls.

Trace-only comparison finds that the first changes occur before the diagnosed
hairpin: track 3 action 54 changes steer +0.10095 to -0.10474; track 4 action
62 changes -0.11163 to +0.15041. Exact saved-camera prefix probes subsequently
show both have no active pass and no detected current circles; their remaining
post-pass cooldown is 4/5 actions. The initial active-pass hypothesis is not
supported. A near road-depth preference reverses the longer reference's turn
direction in both cases. Track 3 then contacts at actions 98/99 and track 4 at
145. A separate causal follow-up can preserve turn direction while excluding
active passes and current circles. The original diagnosed frames 77/78 keep
the same negative turn direction. No such change or extra lap was run in this
V1 artifact; the fixture probes replay only 55/63 saved actions.

Source SHA256:
`ea34ba8e8d9bb414d2fc28f73a9728f791b58ace40f0d00e20b0bb49a1c42914`.
`arc-guard-v1-lineage.json` binds source, parent, patch, real camera fixtures,
tests, freeze and receipt. No new holdout was opened or official code changed.
