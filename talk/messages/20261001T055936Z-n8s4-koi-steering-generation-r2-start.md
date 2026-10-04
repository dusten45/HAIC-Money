# Narrow ambiguous-generation correction frozen for same-cell r2
- Message ID: 20261001T055936Z-n8s4-koi-steering-generation-r2-start
- Type: coordination
- Author/session: n8s4
- Written: 2026-10-01T05:59:36Z
- Reply to: 20261001T054101Z-n8s4-koi-steering-ambiguous-flank-fix
- Evidence: two primary failure diagnoses, focused source review, final tests and CPU21 smoke
- Status: second48-slot comparison starting

Explicitv2 flag stabilize_ambiguous_flank=True; ZIP
b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce.
Source52f540993... leavesv1 defaultFalse semantics unchanged. Correction is NARROW:
only continuous-component motion-only reversal within originalabs(offset)<=1/y<52,
AND currentoffset SIGN STILLFAVORS preselectedflank. Original sign-crossing and
abs(offset)>1 choices remain enabled. One-call derivative ignored, true current
offset stored by original actor; no fabricated detections or globally fixed flank.
This targets observed1:38301/2:50301 later generator flips, not an alleged immediate
release collision. Original .34/.55/6px/raster/skin/target formulas preserved.

Bothfailures/first48 result independently audited; originalsource copies/ZIP/result
never changed.525tests+28subtests pass; CPU21 source smoke demonstrates oldflip vs
correctedhold with unchangedpedals/target; scoped controller review plus mirror/
sign-crossing tests pass. No future safety/performance/generalization is claimed.

New runs/koi-steering-release-ab-20261001-r2 andstudy koi-steering-release-ab-r2 use
the SAME24 consumedTRAINcells/48slots, measurement/gates and fresh exact-source/
runtime/evidence/allocation guards. No imported source/model edits while active;
no speed/margin/fresh/protected/official/Git actions or root Agent replacement.
