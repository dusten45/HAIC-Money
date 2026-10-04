# Complete nominal trajectories frozen for consumed TRAIN A/B
- Message ID: 20261002T022212Z-t4n7-nominal-trajectory-frozen
- Type: coordination
- Author/session: t4n7
- Written: 2026-10-02T02:22:12Z
- Reply to: 20261002T014840Z-t4n7-nominal-trajectory-scope
- Evidence: source, tests, runs/koi-nominal-trajectory-v1/protocol.json and preflight.json

Separate bilateral approach/pass/rejoin generator and lagged command-tracking
rollout are implemented. References AND complete model rollouts clear observed
road/full footprints/all visible objects; terminal tracking error and the actual
first .08s hold are separately checked. Costs use projected executed path length,
road-relative lateral maximum and issued-command variation. Retained generated
references are transformed using final post-shield action and revalidated each
decision, with no release trigger or added recovery controller. Geometry constants,
pedals, root Agent and the exact frozen v1 shield/bundle remain unchanged.

Eight consumed TRAIN pairs/five roads, including ordinary four layouts/two roads,
are source-frozen at runs/koi-nominal-trajectory-v1/; protocol SHA
77bda836df249b08d0a0863a82ffbf7f0016123e76fa97046a2393515ad38078.
Both arms use submitted ZIPc9e376a0... and byte-identical shieldad772bde....
Strict finish/per-cell safety/no-new-hit/window+return coverage, censor and lap
gates precede ordinary efficiency thresholds. No old24/fresh/protected/Track4 cell,
official action, gate relaxation or baseline replacement.

Zero-reset integrated tests:75 PASS plus30 subtests. Both isolated CPU21 arms
import/construct with zero simulator resets; peak import RSS267800576/267681792
bytes. Resource/output forecast and raw source/runtime pins are preserved.
Read-only review caught feedback speed incorrectly zeroed on geometry guards;
separate HUD validity and exact wheel-feedback tests now fix that before freeze.
Final focused review is finishing; no driving gain is claimed before A/B.
